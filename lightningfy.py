from pytorch_lightning import LightningModule
import torch
from torch import nn
from typing import Optional
from torchmetrics import CalibrationError, Accuracy
import torch.nn.functional as F
from typing import Optional, Literal
from utils import DATASETS_INFO

def lightningfy(cfg: dict,
                model: torch.nn.Module,
                temperature_scaler: Optional[nn.Module]=None):
    return LightningWrapper(cfg, model, temperature_scaler)


class LightningWrapper(LightningModule):
    def __init__(self,
                 cfg: dict, 
                 model: torch.nn.Module,
                 temperature_scaler: Optional[nn.Module]=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg['training']['lr']
        # self.train_batch_size = cfg['training']['train_batch_size']
        # self.eval_batch_size = cfg['training']['eval_batch_size']
        self.momentum = cfg['training']['momentum']
        self.weight_decay = cfg['training']['weight_decay']
        self.loss_fn = cfg['training']['loss_fn']
        if self.loss_fn == 'cross_entropy':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif self.loss_fn == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError(
                f'Loss function {self.loss_fn} not supported')
        dataset_name = cfg['dataset']['name']
        num_classes = DATASETS_INFO[dataset_name]['num_classes']
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        # self.f1 = F1Score(task='multiclass', 
        #              num_classes=self.cfg['dataset']['num_classes'])
        self.ece = CalibrationError(task='multiclass', num_classes=num_classes)
        self.train_ts_at_end = self.cfg['scaling']['train_ts_at_end']
        self.train_ts = self.cfg['scaling']['require_scaling']
        self.temperature_scaler = temperature_scaler
        self.logits, self.labels = [], []
        # self.save_hyperparameters()
        
    def disable_ts_training(self):
        self.train_ts = False
    
    def enable_ts_training(self, train_ts_at_end: bool=False):
        self.train_ts = True
        self.train_ts_at_end = train_ts_at_end
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss.item())
        self.log('train_acc', self.accuracy(y_hat, y).item(), prog_bar=True)
        # self.log('train_f1', self.f1(y_hat, y).item())
        self.log('train_ece', self.ece(y_hat, y).item(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        # self.log('val_loss', loss.item())
        self.log('val_acc', self.accuracy(logits, y).item())
        # self.log('val_f1', self.f1(logits, y).item())
        self.log('val_ece_running', self.ece(logits, y).item())
        self.logits.extend(logits)
        self.labels.extend(y)
        return loss

    def _train_ts(self, logits, labels):
        self.log('val_ece_raw', self.ece(logits, labels).item())
        if self.temperature_scaler is not None:
                self.temperature_scaler.fit(logits,
                                            labels)
                logits_scaled = self.temperature_scaler(logits)
                ece_after = self.ece(logits_scaled, labels)
                self.log('val_ece_scaled', ece_after.item())
        else:
            raise NotImplementedError('No temperature scaler found')
        del logits, labels, logits_scaled
        
    
    def on_validation_epoch_end(self):
        val_logits = torch.stack(self.logits)
        val_labels = torch.stack(self.labels)
        self.logits.clear()
        self.labels.clear()
        if self.train_ts:
            if not self.train_ts_at_end: 
                # print('train temperature scaling')
                self._train_ts(val_logits, val_labels)
            if self.current_epoch == self.trainer.max_epochs - 1:
                # print('train temperature scaling at the end')
                # self.current_epoch is none when directly calling validate without trainer
                self._train_ts(val_logits, val_labels)
            
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        # self.log('test_loss', self.loss_fn(logits, y).item())
        self.log('test_acc', self.accuracy(logits, y).item())
        # self.log('test_f1', self.f1(logits, y).item())
        self.log('test_ece_running', self.ece(logits, y).item())
        if self.temperature_scaler is not None:
            logits_scaled = self.temperature_scaler(logits)
            self.log('test_ece_scaled', self.ece(logits_scaled, y).item())
            self.log('test_acc_scaled', self.accuracy(logits_scaled, y).item())
            # self.log('test_f1_after', self.f1(logits_scaled, y).item())
  
    def configure_optimizers(self, optimizer: Optional[torch.optim.Optimizer] = None,
                                 **kwargs):
        if optimizer is None:
            optimizer = self.cfg['training']['optimizer']
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr,
                                             momentum=self.momentum,
                                             weight_decay=self.weight_decay)
            elif optimizer == 'adam':
                # Don't recommend using Adam
                # To be implemented: A Federated Version of Adam 
                # Maybe refers to 
                # https://openreview.net/forum?id=LkFG3lB13U5
                optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.weight_decay)
            else:
                raise NotImplementedError(
                f'Optimizer {self.optimizer} not supported')
        else:
            optimizer = optimizer
        return optimizer

class SimpleWrapper(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        optimizer_type: Literal["SGD", "Adam"] = "SGD",
        max_epochs: int = 100,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.max_epochs = max_epochs
        self.val_loss = []
        self.val_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(logits, y),  prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("val_running_loss", loss)
        self.log("val_running_acc", accuracy)
        self.val_loss.append(loss)
        self.val_accuracy.append(accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        avg_acc = torch.stack(self.val_accuracy).mean()
        self.log("val_avg_loss", avg_loss)
        self.log("val_avg_acc", avg_acc)
        self.val_loss.clear()
        self.val_accuracy.clear()
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("test_running_loss", loss, prog_bar=True)
        self.log("test_running_acc", accuracy, prog_bar=True)
        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)
        return {"test_loss": loss, "test_accuracy": accuracy}
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        avg_acc = torch.stack(self.test_accuracy).mean()
        self.log("test_avg_loss", avg_loss)
        self.log("test_avg_acc", avg_acc)
        self.test_loss.clear()
        self.test_accuracy.clear()
    

    def configure_optimizers(self):
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_avg_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }



if __name__ == "__main__":
    from modelfactory import ModelFactory
    import yaml
    model_type = 'resnet20'
    dataset_name = 'cifar10'
    from scaling import TemperatureScaler
    ts = TemperatureScaler()
    cfg = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    factory = ModelFactory(cfg)
    model = factory.create_model()
    model = lightningfy(cfg, model, ts)
    from datapipeline import load_centralized_dataset
    trainset, testset, valset = load_centralized_dataset(dataset_name,
                                                         require_val=True,
                                                         val_portion=0.1,
                                                         require_noise=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, 
                                              num_workers=128)
    valloader = torch.utils.data.DataLoader(valset, batch_size=256, 
                                            num_workers=128)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                             num_workers=128)
    import pytorch_lightning as pl
    logger = pl.loggers.WandbLogger(project='centralized-temperature-scaling',
                                    name=f'demo',
                                    config=cfg)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1, 
                         max_epochs=1, 
                         logger=logger, 
                         inference_mode=False)
    trainer.fit(model, train_dataloaders=trainloader, 
                val_dataloaders=valloader)
    trainer.validate(model, valloader)
    trainer.test(model, testloader)
    # print(model.state_dict())
