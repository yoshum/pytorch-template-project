import os.path as osp
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

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return {
            "loss": loss,
            "log": {"loss/train": loss, "acc/train": accuracy},
            "progress_bar": {"acc/train": accuracy},
        }

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {
            "loss": F.cross_entropy(y_hat, y),
            "acc": self.accuracy(y_hat, y),
            "batch_size": y.size(0),
        }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        loss = self.sample_weighted_average(outputs, "loss")
        acc = self.sample_weighted_average(outputs, "acc")
        logs = {"val_loss": loss, "loss/val": loss, "acc/val": acc}
        return {"log": logs, "progress_bar": {"acc/val": acc}}

    def accuracy(self, scores, y):
        _, prediction = torch.max(scores, dim=1)
        return (prediction == y).sum(dtype=float) / y.size(0)

    def sample_weighted_average(self, outputs, key):
        values = torch.stack([x[key] for x in outputs])
        batch_sizes = torch.tensor(
            [x["batch_size"] for x in outputs], device=values.device
        )
        return (batch_sizes * values).sum() / batch_sizes.sum()

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            MNIST(
                osp.expanduser("~/.cache/torch/datasets"),
                train=True,
                download=True,
                transform=train_transforms(),
            ),
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(
                osp.expanduser("~/.cache/torch/datasets"),
                train=True,
                download=True,
                transform=eval_transforms(),
            ),
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(
                osp.expanduser("~/.cache/torch/datasets"),
                train=True,
                download=True,
                transform=eval_transforms(),
            ),
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

        return parser
