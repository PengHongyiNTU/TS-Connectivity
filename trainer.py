import ray
import wandb
from torch.utils.data import DataLoader, Dataset
from modelfactory import ModelFactory
from typing import Optional
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import torch
import warnings
import os



@ray.remote(max_calls=1)
def normal_local_train( 
                    cfg: dict,
                    epoch: int,
                    client_id: int,
                    global_params: dict,
                    trainset: Dataset,
                    valset: Optional[Dataset],
                    testset: Optional[Dataset],
                    *args, **kwargs
                    ):
        print(f'Start training epoch {epoch} on client {client_id}')
        verbose = cfg['project']['verbose']
        torch.set_float32_matmul_precision('high')
        warnings.filterwarnings("ignore")
        local_model = ModelFactory(cfg).prepare_model()
        print('model created')
        local_model.load_state_dict(global_params)
        print('model loaded')
        wandb_run = wandb.init(project=cfg['project']['name'], 
                                name=f'client_{client_id}', 
                                id=f"{cfg['project']['name']}_{client_id}",
                                resume="allow",
                                group='clients')
        local_logger = WandbLogger(
                project = cfg['project']['name'],
                name = f'client_{client_id}',
                config=cfg,
                run=wandb_run,
                )
        trainloader = DataLoader(
            trainset, 
            batch_size=cfg['training']['train_batch_size'],
            num_workers=cfg['dataset']['num_workers'])
        valloader = DataLoader(
            valset, 
            batch_size=cfg['training']['eval_batch_size'],
            num_workers=cfg['dataset']['num_workers']
            )
        testloader = DataLoader(
            testset, 
            batch_size=cfg['training']['eval_batch_size'],
            num_workers=cfg['dataset']['num_workers']
        )
        num_local_rounds = cfg['federation']['local_rounds']
        if not verbose:
            trainer = Trainer(
                    accelerator='gpu',
                    max_epochs=num_local_rounds,
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    *args, **kwargs)
        else:
            trainer = Trainer(
                    accelerator='gpu',
                    max_epochs=num_local_rounds,
                    logger=local_logger,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    *args, **kwargs) 
        trainer.fit(local_model, trainloader, valloader)
        results = {}
        metrics = ['train_loss', 'train_acc',  'train_ece',
                     'val_acc',  'val_ece_running']
        results['client_id'] = client_id
        results['model_state_dict'] = local_model.model.state_dict() 
        results['ts_state_dict'] = local_model.temperature_scaler.state_dict()
        results['epoch'] = epoch
        for metric in metrics:
            results[metric] = trainer.callback_metrics[metric].item()
        if cfg['scaling']['require_scaling']:
            results['val_ece_raw'] = trainer.callback_metrics['val_ece_raw'].item()
            results['val_ece_scaled'] = trainer.callback_metrics['val_ece_scaled'].item()
        trainer.test(local_model, testloader)
        test_metrics = [ 'test_acc', 'test_ece_running']
        for metric in test_metrics:
            results[metric] = trainer.callback_metrics[metric].item()
        if cfg['scaling']['require_scaling']:
            results['test_ece_scaled'] = trainer.callback_metrics['test_ece_scaled'].item()
            results['test_acc_scaled'] = trainer.callback_metrics['test_acc_scaled'].item()
        local_model.cpu()
        torch.cuda.empty_cache()
        wandb_run.finish()
        return results