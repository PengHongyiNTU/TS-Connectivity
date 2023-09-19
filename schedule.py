import gpustat
import torch
from pytorch_lightning import Trainer
from lightningfy import LightningWrapper, lightningfy
from torch.utils.data import DataLoader
from typing import List
import os



def get_most_available_gpus(num_gpus: int, max_memory: int=10000) -> List[int]:
    """
    Returns the indices of the most available GPUs.
    """
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_stats = sorted(gpu_stats, key=lambda x: x.memory_free, reverse=True)
    gpu_stats = [gpu for gpu in gpu_stats if gpu.memory_free > max_memory]
    gpu_indices = [gpu.index for gpu in gpu_stats]
    print(f'Available GPUs: {gpu_indices}')
    return gpu_indices[:num_gpus]


class MemoryEstimator:
    def __init__(self, 
                 lightning_model: LightningWrapper,
                 gpu_id: int=0):
        self.model = lightning_model
        self.gpu_id = gpu_id
        self.trainer = Trainer(accelerator='gpu', 
                               devices=[gpu_id],
                               fast_dev_run=True,
                               max_epochs=1, 
                               log_every_n_steps=1,
                               logger=False, 
                               enable_model_summary=False)
        torch.set_float32_matmul_precision('high')
        
        
    def __call__(self, trainloader: DataLoader, valloader: DataLoader):
        print('Starting memory usage estimation...')
        torch.cuda.set_device(self.gpu_id)
        torch.cuda.reset_peak_memory_stats(device=self.gpu_id)
        try:
            self.trainer.fit(self.model, 
                             train_dataloaders=trainloader,
                             val_dataloaders=valloader)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Model too large for GPU {self.gpu_id}")
            raise e
        peak_memory_usage = torch.cuda.max_memory_allocated(device=self.gpu_id) / 1024**2  # Convert bytes to MiB
        return peak_memory_usage
        # gpu_stats_before = gpustat.GPUStatCollection.new_query()
        # memory_used_before = gpu_stats_before[0].memory_used  # assuming you're using GPU 0
        # self.trainer.fit(self.model, 
        #                  train_dataloaders=trainloader,
        #                  val_dataloaders=valloader, 
        #                  )
        # gpu_stats_after = gpustat.GPUStatCollection.new_query()
        # memory_used_after = gpu_stats_after[0].memory_used  # assuming you're using GPU 0
        # peak_memory_usage = memory_used_after - memory_used_before
        # return peak_memory_usage

        
        
def estimate_memory_usage(model: LightningWrapper, 
                          loader: DataLoader) -> int:
    """
    Returns the estimated memory usage of the model.
    """
    peak_memory_usage = MemoryEstimator(model)(loader)
    return peak_memory_usage


def estimate_capacity(
    num_gpus: int, 
    max_memory: int, 
    model: LightningWrapper, 
    trainloader: DataLoader,
    valloader: DataLoader):
    """
    Schedules the training of the model.
    Firstly, get the most available GPUs.
    Then, estimate the memory usage of the model.
    Thirdly, calculate how many process can be run in parallel.
    num_process equals to the sum of selected GPUs' available memory.
    divided by the estimated memory usage of the model.
    """
    gpu_ids = get_most_available_gpus(num_gpus, max_memory)
    print('Starting GPU scheduling...')
    print(f'Most Available GPUs: {gpu_ids}')
    print(f'Using {gpu_ids[0]} for a memory usage estimation...')
    peak_memory_usage = MemoryEstimator(model, gpu_id=gpu_ids[0])(trainloader, valloader)
    # intentionally add 200 MiB surplus to avoid memory overflow
    print(f'Estimated memory usage is {peak_memory_usage} MiB.')
    peak_memory_usage = peak_memory_usage + 200
    print(f'Adding 200Mb to avoid overflow {peak_memory_usage} MiB.')
    max_capacity = {}
    for gpu_id in gpu_ids:
         available_gpu_memory = gpustat.GPUStatCollection.new_query()[gpu_id].memory_free
         num_process = available_gpu_memory // (peak_memory_usage+1e-6)
         max_capacity[gpu_id] = num_process
    if min(max_capacity.values()) < 1:
        raise ValueError('The model is too large to fit in the GPU.')
    print(f'The max capacity is: {max_capacity}')
    return max_capacity
        
    
    
if __name__ == "__main__":
    import yaml
    from modelfactory import ModelFactory
    from datapipeline import load_centralized_dataset
    from scaling import TemperatureScaler
    ts = TemperatureScaler()
    cfg_dict = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    model_type = cfg_dict['model']['type']
    dataset_name = cfg_dict['dataset']['name']
    model = ModelFactory(cfg_dict).create_model()
    # model = lightningfy(model, cfg_dict)
    model = lightningfy(cfg_dict, model, ts)
    trainset, _, valset = load_centralized_dataset(dataset_name,
                                              require_val=True,
                                              val_portion=0.1,
                                              require_noise=False)
    train_batch_size = cfg_dict['training']['train_batch_size']
    eval_batch_size = cfg_dict['training']['eval_batch_size']
    trainloader = DataLoader(trainset, batch_size=train_batch_size, num_workers=4)
    valloader = DataLoader(valset, batch_size=eval_batch_size, num_workers=4)
    estimate_capacity(3, 10000, model, trainloader, valloader)
    
    