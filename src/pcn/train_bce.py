

import lightning
import torch
import bpc_e
import wandb 

from datamodules import CIFAR10, CIFAR100, TinyImageNet
from pcn.datasets import get_CIFAR10_dataloaders
from get_arch import get_architecture_bpc

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from bpc_e import PCE, PC_States, PCESkipConnection
import torch.nn as nn

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback



def train_model(logger, run_config):
    # Make sure to always generate the *exact* same datasets & batches
    lightning.seed_everything(run_config["seed"], workers=True)

    # 1: load dataset as Lightning DataModule
    batch_size = run_config["batch_size"]
    if run_config["dataset"] == "CIFAR10":
        datamodule = CIFAR10(batch_size, is_test=run_config["is_test"]) 
    

    # 2: Setup trainer

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[EarlyStopping(monitor="E_total", patience=100000, mode="min", verbose=True)], #
        max_epochs=run_config["nm_epochs"],
        inference_mode=False,  # inference_mode would interfere with the state backward pass
        limit_predict_batches=1,  # enable 1-batch prediction
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,  # Only run validation after every 25 epochs
    )

    # 3: Get architecture that belongs to this dataset
    architecture = get_architecture_bpc(dataset="CIFAR10", model_name=run_config["model"], activation=run_config["act_fn"])

    # 4: Initiate model and train it
    datamodule.setup("fit")
    pc = PCE(
        architecture,
        iters=run_config["iters"],
        e_lr=run_config["e_lr"],
        w_lr=run_config["w_lr"],
        alpha_up=config["alpha_up"],
        alpha_down=config["alpha_down"],
        output_loss_scale=run_config["output_loss_scale"],
        weight_decay=run_config["w_decay"],
        nm_batches=len(datamodule.train_dataloader()),
        nm_epochs=run_config["nm_epochs"],
    )
    if run_config["load_path"] is not None:
        pc = PCE.load_from_checkpoint(
            run_config["load_path"],
            architecture=architecture,   # pass objects not serializable as hparams
        )


    trainer.fit(pc, datamodule=datamodule)

    # 5: Test results
    trainer.test(pc, datamodule=datamodule)

    # 6: Release all CUDA memory that you can
    pc = None
    trainer = None
    lightning.pytorch.utilities.memory.garbage_collection_cuda()
    torch.cuda.empty_cache()
















if __name__ == "__main__":
    config = {
        "seed": 0,
        "batch_size": 256,
        "nm_epochs": 25,
        "iters": 5,
        "e_lr": 0.001,
        "w_lr": 0.0002684922681018005,
        "w_decay": 0.00000922776551566587,
        "output_loss_scale": 1.0,
        "model": "VGG5",    
        "act_fn": "gelu",
        "dataset": "CIFAR10",
        "is_test": False,   
        "alpha_up": 1.0,
        "alpha_down": 1e-8,   
        "load_path": None, #"checkpoints/tiny-imagenet-VGG16-24-69.69163.ckpt", #"checkpoints/CIFAR10-VGG5-23-23.24405.ckpt",  # if not None, load weights from this path     
        "save_checkpoints": False,
    }

    wandb.init(project="bpc" )
    logger = WandbLogger(project="bpc" , mode="online")
    logger_config = logger.experiment.config

    # overwrite config with logger config if it exists
    for key, value in logger_config.items(): 
        config[key] = value
    logger.experiment.config.update(config) # update wandb config with the full config

    train_model(logger, config)
    wandb.finish()