import warnings
import torch
import argparse
from modelfactory import ModelFactory
from pytorch_lightning import Trainer
from datapipeline import load_dataset
import wandb
from pytorch_lightning.loggers import WandbLogger
from lightningfy import SimpleWrapper
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import numpy as np
from utils import DATASETS_INFO



def centralized_train(dataset_name,
          model_name,
          max_epochs,
          batch_size,
          lr,
          momentum,
          weight_decay,
          optimizer_type,
          enable_wandb_logging=False,
          device="cpu"):
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    cfg = {
        "dataset": {
            "name": dataset_name,
            "require_val": True,
            "val_portion": 0.1,
        },
        "model": {"type": model_name},
        "noise": {"require_noise": False},
    }
    num_classes = DATASETS_INFO[dataset_name]["num_classes"]
    factory = ModelFactory(cfg)
    model = factory.create_model()
    trainset, testset, valset = load_dataset(cfg)
    print(f"Dataset: {dataset_name}, Model: {model_name}")
    print(f'Number of training examples: {len(trainset)}')
    print(f'Number of validation examples: {len(valset)}')
    print(f'Number of test examples: {len(testset)}')
    hyperparameters = {
        "lr": lr,
        'batch_size': batch_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "optimizer_type": optimizer_type,
        "max_epochs": max_epochs,
    }
    if enable_wandb_logging == True:
        run = wandb.init(
            project="local-baseline",
            config=hyperparameters,
            group=f"{dataset_name}-{model_name}",
        )
        logger = WandbLogger(run=run)
    else:
        logger = None
    ln_model = SimpleWrapper(
        model, 
        num_classes, 
        lr, 
        momentum, 
        weight_decay, 
        optimizer_type, 
        max_epochs
    ) 
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_avg_acc',
        dirpath='models/centralized_baselines',
        filename=f'{dataset_name}-{model_name}',
        save_top_k=1,
        mode='max'
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if device != 'cpu' else None,
        devices=[int(args.device)] if device != 'cpu' else None,
        enable_model_summary=True,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(ln_model, train_loader, val_loader)
    test_results = trainer.test(ln_model, test_loader)
    final_test_loss = test_results[0]["test_avg_loss"]
    final_test_acc = test_results[0]["test_avg_acc"]
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    if enable_wandb_logging:
        run.finish()

    print("Training and testing completed!")
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Local Training Script')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (relevant if using SGD)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--optimizer_type', type=str, choices=["SGD", "Adam"], default="SGD", help='Optimizer type')
    parser.add_argument('--enable_wandb_logging', action='store_true', help='Enable logging with Weights & Biases')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='0', help='Device to run training on. "cpu" or GPU index as a string (e.g., "0", "1").')
    args = parser.parse_args()
    set_seed(args.seed)
    centralized_train(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer_type,
        enable_wandb_logging=args.enable_wandb_logging,
        device=args.device
    )
    