import torch



UNIFORM_SIZE = (32, 32)

TRANSFORMATIONS = {
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "emnist": {
        "mean": (0.1751,),
        "std": (0.3267,),
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "fashionmnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
    },
    "tinyimagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}



DATASETS_INFO = {
    "mnist": {"num_classes": 10, "num_channels": 1, },
    "emnist": {"num_classes": 47, "num_channels": 1},
    "fashionmnist": {"num_classes": 10, "num_channels": 1},
    "svhn": {"num_classes": 10, "num_channels": 3},
    "cifar10": {"num_classes": 10, "num_channels": 3},
    "cifar100": {"num_classes": 100, "num_channels": 3},
    "tinyimagenet": {"num_classes": 200, "num_channels": 3}
}

SUPPORTED  = {
            'mnist': ['mlp', 'cnn'],
            'emnist': ['mlp', 'cnn'],
            'fashionmnist': ['mlp', 'cnn'],
            'cifar10': ['resnet20', 'resnet56', 'resnet110'],
            'cifar100': ['resnet20', 'resnet56', 'resnet110'],
            'svhn': ['resnet20', 'resnet56', 'resnet110'],
            'tinyimagenet': ['resnet20', 'resnet56', 'resnet110'],
        }

MAX_EPOCHS = {
    'mnist': 50,
    'emnist': 50,
    'fashionmnist': 50,
    'cifar10': 200,
    'cifar100': 200,
    'tinyimagenet': 300,
    'svhn': 100,
}


def check_tensors_require_grad(obj):
    if isinstance(obj, tuple) and isinstance(obj[1], torch.Tensor):
        if obj[1].requires_grad:
            return [obj[0]]
        else:
            return []
    elif isinstance(obj, list):
        tensors_require_grad = []
        for item in obj:
            tensors_require_grad.extend(check_tensors_require_grad(item))
        return tensors_require_grad
    elif isinstance(obj, dict):
        tensors_require_grad = []
        for k, v in obj.items():
            tensors_require_grad.extend(check_tensors_require_grad((k, v)))
        return tensors_require_grad
    else:
        return []
    
def check_tensors_are_leaf(train_task):
    non_leaf_tensors = []
    for item in train_task:
        if isinstance(item, dict):  # For global_params
            for name, param in item.items():
                if isinstance(param, torch.Tensor) and not param.is_leaf:
                    non_leaf_tensors.append(name)
        elif isinstance(item, torch.utils.data.Subset):  # For trainset and valset
            for x, y in item:
                if isinstance(x, torch.Tensor) and not x.is_leaf:
                    non_leaf_tensors.append("x in Subset")
                if isinstance(y, torch.Tensor) and not y.is_leaf:
                    non_leaf_tensors.append("y in Subset")
    return non_leaf_tensors


    
    
