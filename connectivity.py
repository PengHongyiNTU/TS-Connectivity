import ray
import warnings
from datapipeline import load_centralized_dataset
from modelfactory import ModelFactory
from lightningfy import SimpleWrapper
import wandb
from centralized import run_experiments
from split import LDASplitter
import torch
from noise import uniform_mix_C, flip_labels_C, flip_labels_C_two, NoisyDatasetWrapper



RECOMMENDED_HYPERPARAMETERS = {
    'mnist': {
        'model_name': 'cnn',
        'max_epochs': 50,
        'batch_size': 256,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'optimizer_type': 'SGD',
        'num_classes': 10,
    },
    'cifar10': {
        'model_name': 'resnet20',
        'max_epochs': 200,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'optimizer_type': 'SGD',
        'num_classes': 10,
        },
    'cifar100': {
        'model_name': 'resnet56',
        'max_epochs': 200,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'optimizer_type': 'SGD',
        'num_classes': 100,
    }
    
}



    

    

def prepare_modes_in_parallel():
    ray.init(num_gpus=1)
    run_experiments_parallel = ray.remote(num_gpus=0.2)(run_experiments)
    tasks = []
    dataset_names = ['mnist', 'cifar10', 'cifar100']
    for dataset_name in dataset_names:
        # print(dataset_name)
        model_name = RECOMMENDED_HYPERPARAMETERS[dataset_name]['model_name']
        factory = ModelFactory({'dataset': {'name': dataset_name}, 
                                'model': {'type': model_name}})
        model = factory.create_model()
        # Centralized
        mode = 'centralized'
        trainset, testset, valset = load_centralized_dataset(
            dataset_name=dataset_name,
            require_val=True,
            val_portion=0.1,
            require_noise=False,
            noise_type=None,
            noise_ratio=0.0)
        # print('len(trainset):', len(trainset))
        hyperparameters = RECOMMENDED_HYPERPARAMETERS[dataset_name]
        passing_args = (model, 
                        trainset, 
                        testset, 
                        valset, 
                        hyperparameters, 
                        True, 
                        '0', 
                        'connectivity', 
                        f'{dataset_name}-{model_name}-{mode}')
        tasks.append(run_experiments_parallel.remote(*passing_args))
        # Non-IID
        mode = 'niid'
        alpha = 0.6 
        num_clients = 2
        spliter = LDASplitter(num_clients, alpha)
        train_mapping = spliter.split(trainset, train=True)
        for i, trainset_indices in enumerate(train_mapping['clients_idx'].values()):
            sub_trainset = torch.utils.data.Subset(trainset, trainset_indices)
            passing_args = (model, sub_trainset, testset, valset, 
                            hyperparameters, True, '0', 
                            'connectivity', 
                            f'{dataset_name}-{model_name}-{mode}-{i}')
            tasks.append(run_experiments_parallel.remote(*passing_args))
            # print('len(trainset):', len(sub_trainset))
        # Noisy 
        mode = 'noisy'
        noise_ratio = 0.3
        noise_matrix = uniform_mix_C(noise_ratio, 
                                     num_classes=hyperparameters['num_classes'])
        noisy_trainset = NoisyDatasetWrapper(trainset, noise_matrix)
        # print('len(trainset):', len(noisy_trainset))
        passing_args = (model, noisy_trainset, testset, valset,
                        hyperparameters, True, '0',
                        'connectivity', f'{dataset_name}-{model_name}-{mode}')
        tasks.append(run_experiments_parallel.remote(*passing_args))
        
    
    # Wait for all tasks to finish
    results = ray.get(tasks)
        
        
if __name__ == '__main__':
    prepare_modes_in_parallel()
        
    
    
    
    
    
    
    
    
    
    
    
