from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Callable, List, Tuple, Dict, Optional
import torch
from aggregation import FedAVG
from selection import RandomSelection
from schedule import estimate_capacity
from pytorch_lightning.loggers import WandbLogger
from datapipeline import split_dataset
import numpy as np
import wandb
import math
import pandas as pd
import yaml
from modelfactory import ModelFactory
from datapipeline import load_dataset
from trainer import normal_local_train
import os
import warnings
import ray

class FederatedSimulator:
    def __init__(self,
                 cfg: dict,
                 trainset: Dataset,
                 valset: Optional[Dataset],
                 testset: Dataset,
                 local_train_fn: Callable,
                 aggregation: Optional[Callable] = None,
                 ts_aggregation: Optional[Callable] = None,
                 client_selection: Optional[Callable] = None):
        super().__init__()
        self.cfg = cfg
        ray.init(num_cpus=128, num_gpus=self.cfg['project']['num_gpus'])
        self.factory = ModelFactory(cfg)
        self.model = self.factory.prepare_model()
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.local_train_fn = local_train_fn
        self.aggregation = aggregation if aggregation is not None else FedAVG()
        self.ts_aggregation = ts_aggregation if ts_aggregation is not None else FedAVG()
        self.client_selection = client_selection if client_selection is not None else RandomSelection()
        project_name = self.cfg['project']['name']
        self.wandb_run = wandb.init(project=project_name, 
                                 config=self.cfg, 
                                 group='server', 
                                 name="server", 
                                 id=f"{project_name}_server",
                                 resume="allow")
        self.logger = WandbLogger(run=self.wandb_run)
        # Initialization 
        self._set_seed(self.cfg['project']['seed'])
        self.gpu_capacity = self._check_gpu_capacity()
        # self.gpu_capacity = {0: 20, 1: 20}
        self.train_idx_mapping, self.test_idx_mapping, self.val_idx_mapping = self._get_data_map()
        self.clients_weights = {
            client_id: len(indices) for client_id, indices in
            self.train_idx_mapping['clients_idx'].items()
        }
        # print(self.train_idx_mapping)
        self.wandb_table = None
        print('Simulator initialzied')
        # The main problem 
        # acturally, it seems quite important to use only one GPU to validate 
        # As the multi-GPU validation will spawn subprocess  
        self.global_trainer = Trainer(
            accelerator="gpu",
            devices=[0],
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=self.logger, 
            max_epochs=1, 
            inference_mode=False)
        self.saving_path = f'./models/{project_name}/'
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

    def run(self):
        num_epochs = self.cfg['federation']['num_epochs']
        for epoch in range(num_epochs):
            epoch = epoch + 1
            clients_per_round = self.cfg['federation']['clients_per_round']
            num_clients = self.cfg['federation']['num_clients']
            clients_list = list(range(num_clients))
            selected_clients = self.client_selection(clients_list=clients_list,
                                                     clients_per_round=clients_per_round
                                                     )
            global_params = self.model.state_dict()
            results_list_ids_batches = self.train(epoch, 
                                              global_params, 
                                              selected_clients)
            results_list_batches = [ray.get(batch_ids) for 
                                    batch_ids in 
                                    results_list_ids_batches]
            results_df, local_params_list, ts_params_list = self._convert_results(results_list_batches)
            stats_df = self._analyze(results_df)
            self._log_results(results_df, stats_df)
            selected_weights = [self.clients_weights[client_id] for client_id in selected_clients]
            # to be implemented 
            # seperate parameters of model and ts
            aggregated_params = self.aggregation(local_params_list, 
                                                 clients_weights=selected_weights)
            aggregated_ts_params = self.ts_aggregation(ts_params_list,
                                                       clients_weights=selected_weights)
            self.model.model.load_state_dict(aggregated_params)
            self.model.temperature_scaler.load_state_dict(aggregated_ts_params)
            torch.save(self.model, f'{self.saving_path}/model_{epoch}.pt')
            validation_results = self.validation()
            self.wandb_run.log(validation_results)
            test_results = self.test()
            self.wandb_run.log(test_results)
            # print(f'Test results: {test_results}')

    def train(self,
              epoch: int,
              global_params: dict,
              selected_clients: List[int]):
        with torch.no_grad():
            print(f'Epoch {epoch}: Training on {selected_clients}')
            train_tasks = []
            batches = self._auto_schedule(selected_clients)
            bathed_tasks_ptrs = []
            for batch in batches:
                for client_id in batch:
                    train_indices = self.train_idx_mapping['clients_idx'][client_id]
                    val_indices = self.val_idx_mapping['clients_idx'][client_id]
                    test_indices = self.test_idx_mapping['clients_idx'][client_id]
                    trainset = Subset(self.trainset, train_indices)
                    valset = Subset(self.valset, val_indices)
                    testset = Subset(self.testset, test_indices)
                    gpu_fraction = 1 / sum(self.gpu_capacity.values())
                    gpu_fraction = math.ceil(round(gpu_fraction, 2) * 100) / 100
                    task = [self.cfg,
                            epoch,
                            client_id,
                            global_params,
                            trainset,
                            valset,
                            testset]
                    print(f'Client {client_id} is using GPU {gpu_fraction}')
                    train_tasks.append(
                        self.local_train_fn.options(
                            num_gpus=gpu_fraction).remote(*task)
                    )
            bathed_tasks_ptrs.append(train_tasks)
            return bathed_tasks_ptrs

    def _convert_results(self, results_list_batches: List[List[Dict]]) -> Tuple[pd.DataFrame, List[Dict]]:
        flatten_results_list = [result for sublist in results_list_batches for result in sublist]
        local_params_list = [result.pop('model_state_dict') for result in flatten_results_list]
        ts_params_list = [result.pop('ts_state_dict') for result in flatten_results_list]
        df = pd.DataFrame(flatten_results_list)
        return df, local_params_list, ts_params_list

    def _compute_statistics(self, df, metric):
        avg = df[metric].mean()
        var = df[metric].var() if len(df[metric]) > 1 else 0
        mean = df[metric].mean()
        return avg, var, mean

    def _analyze(self, results_df: pd.DataFrame):
        print('Analyzing returned results')
        metrics = ['train_loss', 'train_acc',  'train_ece',
                   'val_acc', 'val_ece_running', 
                   'test_acc', 'test_ece_running']
        stats_dict = {}
        for metric in metrics:
            avg, var, ptp = self._compute_statistics(results_df, metric)
            # Log metrics with epoch as x-axis
            stats_dict[f'clients_{metric}_mean'] = avg
            stats_dict[f'clients_{metric}_var'] = var
            stats_dict[f'clients_{metric}_ptp'] = ptp

        if self.cfg['scaling']['require_scaling']:
            clients_val_ece_raw = results_df['val_ece_raw'].mean()
            clients_val_ece_scaled = results_df['val_ece_scaled'].mean()
            clients_test_ece_scaled = results_df['test_ece_scaled'].mean()
            clients_test_acc_scaled = results_df['test_acc_scaled'].mean()
            stats_dict['clients_val_ece_raw'] = clients_val_ece_raw
            stats_dict['clients_val_ece_scaled'] = clients_val_ece_scaled
            stats_dict['clients_test_ece_scaled'] = clients_test_ece_scaled
            stats_dict['clients_test_acc_scaled'] = clients_test_acc_scaled
        self.wandb_run.log(stats_dict)
            
        stats_df = pd.DataFrame([stats_dict])
        return stats_df

    def _log_results(self, results_df: pd.DataFrame,
                     stats_df: pd.DataFrame):
        if self.wandb_table is None:
            self.history_df = results_df
            self.history_stats_df = stats_df
            self.wandb_table = wandb.Table(dataframe=self.history_df)
            self.wandb_stats_table = wandb.Table(dataframe=self.history_stats_df)
        else:
            self.history_df = pd.concat([self.history_df, results_df],
                                        axis=0,
                                        ignore_index=True)
            # print(self.history_df)
            self.history_stats_df = pd.concat([self.history_stats_df, stats_df],
                                              axis=0,
                                              ignore_index=True)
            # print(self.history_stats_df)                             
            # for results in results_df.values.tolist():
            #     self.wandb_table.add_data(*results)
            # for stats in stats_df.values.tolist():
            #     self.wandb_stats_table.add_data(*stats)
        new_table = wandb.Table(dataframe=self.history_df)
        new_stats_table = wandb.Table(dataframe=self.history_stats_df)
        self.wandb_run.log({'hitory': new_table})
        self.wandb_run.log({'history_stats': new_stats_table})

    def test(self):
        if self.test_idx_mapping['global_idx'] is not None:
            global_test_indices = self.test_idx_mapping['global_idx']
            global_testset = Subset(self.testset, global_test_indices)
            testloader = DataLoader(
                global_testset,
                batch_size=self.cfg['training']['eval_batch_size'],
                num_workers=self.cfg['dataset']['num_workers']
            )
            trainer = self.global_trainer
            trainer.test(self.model, testloader)
            metrics_mapping = {
                'test_acc': 'server_test_acc',
                'test_ece_running': 'server_test_ece',                
            }
            if self.model.temperature_scaler is not None:
                metrics_mapping['test_ece_scaled'] = 'server_test_ece_scaled'
                metrics_mapping['test_acc_scaled'] = 'server_test_acc_scaled'
            test_results = {}
            for metric in metrics_mapping.keys():
                test_results[metrics_mapping[metric]] = trainer.callback_metrics[metric].item()
            return test_results

    def validation(self):
        self.model.eval()
        if self.val_idx_mapping['global_idx'] is not None:
            global_val_indices = self.val_idx_mapping['global_idx']
            global_valset = Subset(self.valset, global_val_indices)
            valloader = DataLoader(
                global_valset,
                batch_size=self.cfg['training']['eval_batch_size'],
            )
            trainer = self.global_trainer
            trainer.validate(self.model, valloader)
            val_results = {}
            metrics_mapping  = {
                'val_acc': 'server_val_acc',
                'val_ece_running': 'server_val_ece_running',
            }
            if self.model.temperature_scaler is not None:
                metrics_mapping['val_ece_raw'] = 'server_val_ece_raw'
                metrics_mapping['val_ece_scaled'] = 'server_val_ece_scaled'
            for metric in metrics_mapping.keys():
                val_results[metrics_mapping[metric]] = trainer.callback_metrics[metric].item()
            return val_results
        
        

    def _get_data_map(self):
        train_idx_mapping, test_idx_mapping, val_idx_mapping = split_dataset(
            cfg=self.cfg,
            trainset=self.trainset,
            testset=self.testset,
            valset=self.valset
        )
        return train_idx_mapping, test_idx_mapping, val_idx_mapping

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _auto_schedule(self, selected_clients: List[int]):
        clients_per_round = len(selected_clients)
        gpu_capacity = self.gpu_capacity
        if clients_per_round < sum(gpu_capacity.values()):
            print(f'{clients_per_round} are less than the total capacity of GPUs {sum(gpu_capacity.values())}')
            return [selected_clients]
        else:
            print(f'{clients_per_round} are more than the total capacity of GPUs {sum(gpu_capacity.values())}')
            total_capacity = sum(gpu_capacity.values())
            batches = [selected_clients[i:i + total_capacity] for i in range(0, len(selected_clients), total_capacity)]
            print(f'Will run {len(batches)} batches of simulations')
            return batches

    def _check_gpu_capacity(self):
        train_batch_size = self.cfg['training']['train_batch_size']
        eval_batch_size = self.cfg['training']['eval_batch_size']
        num_workers = self.cfg['dataset']['num_workers']
        trainloader = DataLoader(self.trainset, batch_size=train_batch_size,
                                 num_workers=num_workers)
        valloader = DataLoader(self.valset, batch_size=eval_batch_size,
                               num_workers=num_workers)
        prefer_num_gpus = self.cfg['project']['num_gpus']
        gpu_capacity = estimate_capacity(
            prefer_num_gpus, 10000, self.model, trainloader, valloader)
        return gpu_capacity

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    trainset, testset, valset = load_dataset(cfg)
    simulator = FederatedSimulator(
        cfg=cfg,
        trainset=trainset,
        valset=valset,
        testset=testset,
        local_train_fn=normal_local_train)
    simulator.run()
