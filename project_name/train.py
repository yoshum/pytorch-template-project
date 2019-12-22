"""
This file runs the main training/val loop, etc...
using Lightning Trainer
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser

import project_name.pl_modules as pl_modules


def main(hparams):
    # init module
    model = pl_modules.MNISTLightningModule(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--hash", type=str, default=None)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = pl_modules.MNISTLightningModule.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)
