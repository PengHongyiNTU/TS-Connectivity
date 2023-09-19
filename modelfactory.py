import torch
from torchvision import models
from torch import nn
import yaml
import torch.nn.functional as F
from scaling import TemperatureScaler, MLPScalar
from lightningfy import LightningWrapper
from utils import DATASETS_INFO, SUPPORTED
from resnet import resnet20, resnet56, resnet110


class MLP(nn.Module):
    def __init__(self, input_size=1024, hidden_sizes=[512, 512], num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], num_classes))  # Corrected parentheses placement
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x 

class CNN(nn.Module):
    def __init__(self, num_channels=3, hidden_sizes=[32, 64, 128], 
                 num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(num_channels, hidden_sizes[0], kernel_size=3, stride=1, padding=1))
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Corrected 'kernal' to 'kernel'
            self.layers.append(nn.Conv2d(hidden_sizes[i], hidden_sizes[i+1], kernel_size=3, stride=1, padding=1))
        self.layers.append(nn.ReLU())  # Corrected 'lauyers' to 'layers'
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



def convert_bn_to_gn(module, num_groups):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            return torch.nn.GroupNorm(num_groups, module.num_features,
                                      eps=module.eps, affine=module.affine)
        else:
            for name, child in module.named_children():
                module.add_module(name, convert_bn_to_gn(child, num_groups))
            return module
        
        
class ModelFactory:
    def __init__(self, cfg:dict):
        self.cfg = cfg
        self.supported = SUPPORTED
        self.dataset_name = cfg['dataset']['name']
        self.model_type = cfg['model']['type']
        
    
    def prepare_model(self, *args, **kwargs):
        model = self.create_model()
        if self.cfg['scaling']['require_scaling']:
            scaler = self.create_ts_model()
        else:
            scaler = None
        return LightningWrapper(self.cfg, model, scaler, *args, **kwargs)
        
    
    
    def create_ts_model(self):
        ts_type = self.cfg['scaling']['type']
        lr = self.cfg['scaling']['lr']
        max_iter = self.cfg['scaling']['max_iter']
        num_classes = DATASETS_INFO[self.dataset_name]['num_classes']
        if ts_type == "ts":
            return TemperatureScaler(lr=lr, max_iter=max_iter)
        elif ts_type == "mlp":
            hidden_neurons = self.cfg['scaling']['hidden_neurons']
            return MLPScalar(logits_dims=num_classes,
                             hidden_neurons=hidden_neurons,
                             lr=lr,
                             max_iter=max_iter)
    
    
    def create_model(self):
        dataset_name = self.dataset_name.lower()
        model_type = self.model_type.lower()
        if dataset_name not in self.supported.keys():
            raise ValueError(f'Unknown dataset {dataset_name}')
        elif model_type not in self.supported[dataset_name]:
            raise ValueError(f'Not supported model {model_type}')
        num_classes = DATASETS_INFO[dataset_name]['num_classes']
        num_channels = DATASETS_INFO[dataset_name]['num_channels']
        if model_type == 'mlp':
           model = MLP(num_classes=num_classes)  # Here you need to pass in the relevant parameters
        elif model_type == 'cnn':
            model = CNN(num_classes=num_classes, num_channels=num_channels)  # Here you need to pass in the relevant parameters
        elif model_type == 'resnet20':
            model = resnet20(num_classes=num_classes)
        elif model_type == 'resnet56':
            model = resnet56(num_classes=num_classes)
        elif model_type == 'resnet110':
            model = resnet110(num_classes=num_classes)
        # elif model_type == 'densenet121':
        #     model = models.densenet121(weights=None)
        #     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        # elif model_type == 'resnext50':
        #     model = models.resnext50_32x4d(weights=None)
        #     model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f'Unsupported model type {model_type}')   
        return model  # Return the model

if __name__ == "__main__":
    import yaml 
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    factory = ModelFactory(config)
    model_mnist = factory.create_model()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model_mnist(dummy_input)
    print(output.shape)
    
    
    