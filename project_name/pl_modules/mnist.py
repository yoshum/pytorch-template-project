import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from argparse import ArgumentParser

import pytorch_lightning as pl

from project_name.models import build_net
from project_name.datasets import train_transforms, eval_transforms


class MNISTLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(MNISTLightningModule, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.net = build_net(None)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.net(x)
        return {"loss": F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.net(x)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=train_transforms()),
            batch_size=self.hparams.batch_size,
        )

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=eval_transforms()),
            batch_size=self.hparams.batch_size,
        )

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(os.getcwd(), train=True, download=True, transform=eval_transforms()),
            batch_size=self.hparams.batch_size,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--learning_rate", default=0.02, type=float)
        parser.add_argument("--batch_size", default=32, type=int)

        # training specific (for this model)
        parser.add_argument("--max_nb_epochs", default=2, type=int)

        return parser
