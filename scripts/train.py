"""
This file runs the main training/val loop, etc...
using Lightning Trainer
"""
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

import project_name.pl_modules as pl_modules

import logging

logging.basicConfig(level=logging.INFO)


def main(hparams):
    # init module
    model = pl_modules.MNISTLightningModule(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger(save_dir="/workspace/results", name="mnist_logs",),
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--hash", type=str, default=None)

    parser.add_argument("--max_epochs", default=2, type=int)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = pl_modules.MNISTLightningModule.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
