from typing import Tuple, Union
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from split import LDASplitter, IIDSplitter
from typing import Dict, Optional
from noise import uniform_mix_C, flip_labels_C, flip_labels_C_two, NoisyDatasetWrapper
import os
from torchvision.datasets.utils import download_and_extract_archive
from utils import UNIFORM_SIZE, DATASETS_INFO, TRANSFORMATIONS
from sklearn.model_selection import train_test_split


def get_transformations(dataset_name: str):
    mean = TRANSFORMATIONS[dataset_name]["mean"]
    std = TRANSFORMATIONS[dataset_name]["std"]
    transoformation = transforms.Compose(
        [
            transforms.Resize(UNIFORM_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    augmented_transformation = transforms.Compose(
        [
            transforms.Resize(UNIFORM_SIZE),
            *get_augmentations(dataset_name),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transoformation, augmented_transformation


def get_augmentations(dataset_name: str):
    if dataset_name in ['mnist', 'emnist', 'fashionmnist']:
        return [
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    
    elif dataset_name in ['cifar10', 'cifar100', 'svhn', 'tinyimagenet']:
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset(dataset_name:str, 
                transforms, 
                augmented_transforms,
                require_val: bool = True,
                val_portion: float = 0.1):
    if require_val:
        assert 0 < val_portion < 1, "val_portion must be in (0, 1)"
    if dataset_name == "mnist":
        data_class = torchvision.datasets.MNIST
        trainset = data_class(
        root="~/Research/Calibration/data",
        train=True,
        transform=augmented_transforms,
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        train=False,
        transform=transforms,
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                train=True,
                transform=transforms,
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    elif dataset_name == "emnist":
        data_class = torchvision.datasets.EMNIST
        trainset = data_class(
        root="~/Research/Calibration/data",
        train=True,
        transform=augmented_transforms,
        split="balanced",
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        train=False,
        transform=transforms,
        split="balanced",
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                train=True,
                transform=transforms,
                split="balanced",
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    elif dataset_name == "fashionmnist":
        data_class = torchvision.datasets.FashionMNIST
        trainset = data_class(
        root="~/Research/Calibration/data",
        train=True,
        transform=augmented_transforms,
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        train=False,
        transform=transforms,
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                train=True,
                transform=transforms,
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    elif dataset_name == "cifar10":
        data_class = torchvision.datasets.CIFAR10
        trainset = data_class(
        root="~/Research/Calibration/data",
        train=True,
        transform=augmented_transforms,
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        train=False,
        transform=transforms,
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                train=True,
                transform=transforms,
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    elif dataset_name == "cifar100":
        data_class = torchvision.datasets.CIFAR100
        trainset = data_class(
        root="~/Research/Calibration/data",
        train=True,
        transform=augmented_transforms,
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        train=False,
        transform=transforms,
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                train=True,
                transform=transforms,
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    elif dataset_name == "svhn":
        data_class = torchvision.datasets.SVHN
        trainset = data_class(
        root="~/Research/Calibration/data",
        split="train",
        transform=augmented_transforms,
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        split="test",
        transform=transforms,
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                split="train",
                transform=transforms,
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    elif dataset_name == "tinyimagenet":
        data_class = TinyImageNet
        trainset = data_class(
        root="~/Research/Calibration/data",
        train=True,
        transform=augmented_transforms,
    )
        testset = data_class(
        root="~/Research/Calibration/data",
        train=False,
        transform=transforms,
    )
        if require_val:
            val_set = data_class(
                root="~/Research/Calibration/data",
                train=True,
                transform=transforms,
            )
            indices  = list(range(len(trainset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_portion, random_state=42
            )
            trainset = Subset(trainset, train_indices)
            valset = Subset(val_set, val_indices)
        else:
            valset = None
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")
    return trainset, testset, valset
   
        
             



class TinyImageNet:    
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        if download:
            self._check_exists()
        if train:
            root = os.path.join(self.root, 'tiny-imagenet-200/train')
            self.dataset = torchvision.datasets.ImageFolder(root, transform=transform)
        else:
            root = os.path.join(self.root, 'tiny-imagenet-200/val')
            self.dataset = torchvision.datasets.ImageFolder(root, transform=transform)
        
    def _check_exists(self):
        if not os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            download_and_extract_archive(url, self.root)
        else:
            pass
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


def load_dataset(cfg: Dict):
    dataset_name = cfg["dataset"]["name"]
    require_val = cfg["dataset"]["require_val"]
    val_portion = cfg["dataset"]["val_portion"]
    require_noise = cfg["noise"]["require_noise"]
    if require_noise:
        noise_type = cfg["noise"]["type"]
        noise_ratio = cfg["noise"]["ratio"]
    else:
        noise_ratio = None
        noise_type = None
    if dataset_name in DATASETS_INFO.keys():
        trainset, testset, valset = load_centralized_dataset(
            dataset_name,
            require_val=require_val,
            val_portion=val_portion,
            require_noise=require_noise,
            noise_type=noise_type,
            noise_ratio=noise_ratio,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return trainset, testset, valset


def split_dataset(
    cfg: Dict, trainset: Dataset, testset: Dataset, valset: Dataset, **kwargs
):
    supported_split_types = ["iid", "lda"]
    split_type = cfg["split"]["type"]
    num_clients = cfg["federation"]["num_clients"]
    if split_type not in supported_split_types:
        raise ValueError(f"split_type must be one of {supported_split_types}")
    else:
        if split_type == "lda":
            assert cfg["split"]["alpha"], f"alpha must be provided for LDA split"
            alpha = cfg["split"]["alpha"]
            splitter = LDASplitter(num_clients, alpha, **kwargs)
        elif split_type == "iid":
            splitter = IIDSplitter(num_clients, **kwargs)
        print("Training Mapping:")
        train_mapping = splitter.split(trainset, train=True)
        require_local_test = cfg["split"]["require_local_test"]
        global_local_ratio = cfg["split"]["global_local_ratio"]
        print("Testing Mapping:")
        test_mapping = splitter.split(
            testset,
            train=False,
            local=require_local_test,
            global_local_ratio=global_local_ratio,
        )
        if valset is not None:
            print("Validation Mapping:")
            require_local_val = cfg["split"]["require_local_val"]
            val_mapping = splitter.split(
                valset,
                train=False,
                local=require_local_val,
                global_local_ratio=global_local_ratio,
            )
        else:
            val_mapping = None
    return train_mapping, test_mapping, val_mapping


def load_centralized_dataset(
    dataset_name: str,
    require_val: bool = False,
    val_portion: Optional[float] = None,
    require_noise: bool = False,
    noise_type: Optional[str] = None,
    noise_ratio: Optional[float] = None,
) -> Tuple[Dataset, Dataset, Union[Dataset, None]]:
    transforms, augmented_transforms = get_transformations(dataset_name)
    trainset, testset, valset = get_dataset(
        dataset_name, transforms, augmented_transforms,
        require_val=require_val, val_portion=val_portion,
    )
    num_classes = DATASETS_INFO[dataset_name]["num_classes"]
    if require_noise:
        print("Applying noise to trainset")
        print(f"noise_type: {noise_type}; noise_ratio: {noise_ratio}")
        assert (
            noise_type in ["uniform", "target1", "target2"] and noise_ratio is not None
        ), "noise_type and noise_ratio must be provided"
        if noise_type == "uniform":
            noise_matrix = uniform_mix_C(noise_ratio, num_classes)
        elif noise_type == "target1":
            noise_matrix = flip_labels_C(noise_ratio, num_classes)
        elif noise_type == "target2":
            noise_matrix = flip_labels_C_two(noise_ratio, num_classes)
        print("noise_matrix: ", noise_matrix)
        trainset = NoisyDatasetWrapper(trainset, noise_matrix)
        return trainset, testset, valset

    else:
        return trainset, testset, valset


if __name__ == "__main__":
    import yaml
    dataset_names = [
        "mnist",
        "emnist",
        "cifar10",
        "cifar100",
        "fashionmnist",
        "svhn",
        "tinyimagenet",
        "emnist",
    ]
    for dataset_name in dataset_names:
        print(f"Loading {dataset_name} dataset...")
        cfg = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
        cfg["dataset"]["name"] = dataset_name
        train_set, test_set, val_set = load_dataset(cfg=cfg)
        train_idx_mapping, test_idx_mapping, val_idx_mapping = split_dataset(
            cfg=cfg,
            trainset=train_set,
            testset=test_set,
            valset=val_set,
        )
        print(f"{dataset_name} dataset loaded successfully!")
        #
