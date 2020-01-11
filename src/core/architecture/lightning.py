import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict
from pytorch_lightning.logging import TestTubeLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST
from architecture.pytorch_extensions import PytorchNet
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from architecture.loss import F2PLoss


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionLightningModel(PytorchNet):
    def __init__(self, hparams=None, resume=True):
        super().__init__()
        if hparams is None:  # Just the model - for forward runs only
            hparams = self.add_model_specific_args([]).parse_args()
        else:
            self.loss = F2PLoss(hparams, device='cuda')  # TODO

        # If you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        self.hparams = hparams
        self._build_model()
        if resume:
            pass  # TODO
        else:
            self._init_model()

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def forward(self, part, template):
        raise NotImplementedError

    def set_loaders(self, loaders):
        # TODO - Consider using a dict instead of list  {'Test':ldr,'Val':ldr,'Train':ldr}
        self.loaders = loaders

    def training_step(self, b, _):

        y_hat = self.forward(b['gt_part_v'], b['tp_v'])
        loss_val = self.loss.compute(b['gt_v'], y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}  # Must be all Tensors
        # TODO - What more do we need to log? Learning Step? Memory Consumption?
        return {
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }

    def validation_step(self, b, _):

        y_hat = self.forward(b)
        loss_val = self.loss(b['gt_v'], y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        return loss_val
        # output = {'val_loss': loss_val,'val_acc': val_acc}

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        return torch.stack(outputs).mean()

        # val_loss_mean = 0
        # val_acc_mean = 0
        # for output in outputs:
        #     val_loss = output['val_loss']
        #
        #     # reduce manually when using dp
        #     if self.trainer.use_dp or self.trainer.use_ddp2:
        #         val_loss = torch.mean(val_loss)
        #     val_loss_mean += val_loss
        #
        #     # reduce manually when using dp
        #     val_acc = output['val_acc']
        #     if self.trainer.use_dp or self.trainer.use_ddp2:
        #         val_acc = torch.mean(val_acc)
        #
        #     val_acc_mean += val_acc
        #
        # val_loss_mean /= len(outputs)
        # val_acc_mean /= len(outputs)
        # tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        # result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        # return result

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=cfg.N_EPOCHS_TO_WAIT_BEFORE_LR_DECAY)
        # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
        #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                            eps=1e-08)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return self.data_loaders[0]

    @pl.data_loader
    def val_dataloader(self):
        return self.data_loaders[1]

    @pl.data_loader
    def test_dataloader(self):
        return self.data_loaders[2]

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser


def train():
    pass


def test():
    pass
