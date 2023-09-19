# This python files aims to find the best hyperparameters
# Hyperparameters to be tuned includes:
# 1. models's local optimizer (SGD, Adam)
# 1. model's optimizer's learning rate
# 2. model's optimizer's momentum
# 3. model's optimizer's weight decay
# 4. training batch size
# Tuned based on the validation accuracy
# ------
# After the best model is found,
# Freeze the best model's parameters
# Tune the temperature scaler's hyperparameters
# Based on the validation ECE
# 5. Scaler's learning rate
# 6. Scaler's max iteration
import ray
from ray.tune import Tuner
import wandb
from ray import tune, air
from modelfactory import ModelFactory
from datapipeline import load_dataset
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from utils import DATASETS_INFO, SUPPORTED, MAX_EPOCHS
from pytorch_lightning.loggers import WandbLogger
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from ray.tune import CLIReporter
import os
import warnings
import pickle
import shutil
from lightningfy import SimpleWrapper


def basemodel_training_worker(
    search_space,
    dataset_name,
    model_name,
    max_epochs=50,
):
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    cfg = {
        "dataset": {
            "name": dataset_name,
            "require_val": True,
            "val_portion": 0.2,
        },
        "model": {"type": model_name},
        "noise": {"require_noise": False},
    }
    factory = ModelFactory(cfg)
    model = factory.create_model()
    trainset, _, valset = load_dataset(cfg)
    batch_size = search_space["batch_size"]
    lr = search_space["lr"]
    momentum = search_space["momentum"]
    weight_decay = search_space["weight_decay"]
    optimizer_type = search_space["optimizer"]
    num_classes = DATASETS_INFO[dataset_name]["num_classes"]
    run = wandb.init(
        project="parameter-tuned-base-model",
        config=search_space,
        group=f"{dataset_name}-{model_name}",
    )
    logger = WandbLogger(run=run)
    lightning_model = SimpleWrapper(
        model,
        num_classes=num_classes,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
    )
    callback = TuneReportCheckpointCallback(
        {
            "accuracy": "val_avg_acc",
            "loss": "val_avg_loss",
        },
        on="validation_end",
        filename="checkpoint",
    )
    run = wandb.init(
        project="parameter-tuned-base-model",
        config=search_space,
        group=f"{dataset_name}-{model_name}",
    )
    logger = WandbLogger(run=run)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        callbacks=[callback],
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=logger,
        gradient_clip_val=0.1,
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    trainer.fit(lightning_model, trainloader, valloader)
    run.finish()


def save_and_clean(results, dataset_name, model_name):
    best_result = results.get_best_result(metric="accuracy", mode="max")
    print(best_result)
    best_config = best_result.config
    print(best_config)
    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        best_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        best_checkpoint = torch.load(best_checkpoint_path)
    state_dict = best_checkpoint["state_dict"]
    to_save = {"config": best_config, "state_dict": state_dict, "result": best_result}
    # Save the pickle file first
    parent_dir = f"tuned/{dataset_name}-{model_name}"
    saving_dir = os.path.join(parent_dir, "best_config.pkl")
    with open(saving_dir, "wb") as f:
        pickle.dump(to_save, f)
    print(f"Saved best config and state dict for {dataset_name}-{model_name}")
    # Now, clean up the directory but ensure you don't delete best_config.pkl
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if (
            os.path.isfile(item_path) and item_path != saving_dir
        ):  # Check to ensure you're not deleting best_config.pkl
            print(f"Deleting file: {item_path}")
            os.remove(item_path)
        elif os.path.isdir(item_path):
            print(f"Deleting directory: {item_path}")
            shutil.rmtree(item_path)


def tune_base_model(
    dataset_name,
    model_name,
    search_space,
    enable_scheduler=False,
    enable_bayesian=False,
    num_samples=10,
):
    max_epochs = MAX_EPOCHS[dataset_name]
    print(f"Tunning for {dataset_name}-{model_name}: Max Epochs: {max_epochs}")
    tune_fn = tune.with_parameters(
        basemodel_training_worker,
        dataset_name=dataset_name,
        model_name=model_name,
        max_epochs=max_epochs,
    )
    resoruces_per_trial = {"cpu": 1, "gpu": 0.5}
    tune_fn = tune.with_resources(
        tune_fn,
        resources=resoruces_per_trial,
    )
    tuner_path = os.path.join(f"tuned/{dataset_name}-{model_name}", "tuner.pkl")
    best_config_path = os.path.join(
        f"tuned/{dataset_name}-{model_name}", "best_config.pkl"
    )
    if os.path.exists(tuner_path):
        print("Tuner founded, loading...")
        print("Try resume tuned...")
        tuner = Tuner.restore(
            path=f"tuned/{dataset_name}-{model_name}",
            trainable=tune_fn,
            restart_errored=True,
            resume_unfinished=True,
        )
        results = tuner.fit()
        save_and_clean(results, dataset_name, model_name)
    elif os.path.exists(best_config_path):
            print("Best config founded, loading...")
            best_config = pickle.load(open(best_config_path, "rb"))
            print(best_config["config"])
            print(best_config["result"].metrics)
            pass
    else:
        print("No tuner founded, start new tuned...")
        if enable_bayesian:
            search = OptunaSearch(metric="accuracy", mode="max")
        else:
            search = BasicVariantGenerator()
        if enable_scheduler:
            scheduler = ASHAScheduler(
                        time_attr="training_iteration",
                        max_t=max_epochs,
            )
        else:
            scheduler = FIFOScheduler()
        reporter = CLIReporter(
            parameter_columns=["lr", "momentum", "weight_decay", "batch_size"],
            metric_columns=["loss", "accuracy", "training_iteration"],
            )
        tuner = tune.Tuner(
                    trainable=tune_fn,
                    tune_config=tune.TuneConfig(
                        metric="loss",
                        mode="min",
                        scheduler=scheduler,
                        num_samples=num_samples,
                        search_alg=search,
                        chdir_to_trial_dir=False,
                    ),
                    run_config=air.RunConfig(
                        name=f"{dataset_name}-{model_name}",
                        progress_reporter=reporter,
                        storage_path="~/Research/Calibration/tuned",
                        checkpoint_config=air.CheckpointConfig(num_to_keep=1),
                    ),
                    param_space=search_space,
                )
        results = tuner.fit()
        save_and_clean(results, dataset_name, model_name)


if __name__ == "__main__":
    ray.init(num_cpus=64, num_gpus=2)
    search_space = {
        "optimizer": tune.choice(["SGD", "Adam"]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "weight_decay": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
    }
    for dataset_name in SUPPORTED.keys():
        model_names = SUPPORTED[dataset_name]
        for model_name in model_names:
            tune_base_model(
                dataset_name=dataset_name,
                model_name=model_name,
                search_space=search_space,
                enable_scheduler=True,
                enable_bayesian=True,
                num_samples=15,
            )
            
