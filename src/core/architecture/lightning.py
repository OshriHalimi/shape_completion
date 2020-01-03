"""
Example template for defining a system
"""
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
from architecture import PytorchNet
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
#This is the easiest/fastest way which loads hyperparameters
#and weights from a checkpoint, such as the one saved by the ModelCheckpoint callback
# pretrained_model = MyLightningModule.load_from_checkpoint(
#     checkpoint_path='/path/to/pytorch_checkpoint.ckpt'
# )


# Train:
# checkpoint_callback
# log_gpu_memory = 'min_max' or 'all'
# track_grad_norm  = 2 # Track L2 norm of the gradient
# fast_dev_run = True - runs full iteration over everything to find bugs
# log_save_interval = 100
# distributed_backend = 'dp', 'ddp', 'ddp2'
# weights_summary = 'full', 'top' , None
# accumulate_grad_batches = 1
# nb_sanity_val_steps = 5 # Sanity checks for validation - Use 0 to optimize
# if not has_checkpoint:
#     nb_sanity_val_stes = 3
#     fast_dev_run = True
# `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
#           if '.ckpt' in name:
#                 epoch = name.split('epoch_')[1]
#                 epoch = int(re.sub('[^0-9]', '', epoch))
#
#                 if epoch > last_epoch:
#                     last_epoch = epoch
#                     last_ckpt_name = name
# EarlyStopping(
#                 monitor='val_loss',
#                 patience=3,
#                 verbose=True,
#                 mode='min'
#             )
# checkpoint_callback = ModelCheckpoint(
#     filepath=os.getcwd(),
#     save_best_only=True,
#     verbose=True,
#     monitor='val_loss',
#     mode='min',
#     prefix=''
# )
# {}_ckpt_epoch_{}.ckpt {prefix}
def worker_init_fn(worker_id):
    random.seed(worker_id + 2434)
    np.random.seed(worker_id + 2434)


def set_random_seed(seed=2434):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
# def make_directory(name):
#     result = Path("result")
#     result.mkdir(exist_ok=True)
#     if name is not None:
#         dir_name = name
#     else:
#         now = datetime.datetime.now()
#         dir_name = datetime.datetime.strftime(now, "%y_%m_%d_%H")
#     log_dir = result / dir_name
#     log_dir.mkdir(exist_ok=True)
#
#     return log_dir

# Restore Training Session:
# logger = TestTubeLogger(
#     save_dir='./savepath',
#     description= 'experiment_name'
#     version=1  # An existing version with a saved checkpoint
# )
# trainer = Trainer(
#     logger=logger,
#     default_save_path='./savepath'
# )
# resume_from_checkpoint - Resume from something specific


class LightningTemplateModel(PytorchNet):
    def __init__(self, hparams,resume=True):
        super().__init__()
        self.hparams = hparams

        # If you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        # build model
        self._build_model()
        if resume:
            pass # Do the Resume
        else:
            self._init_model()

    def _build_model(self):
        raise NotImplementedError
    def _init_model(self):
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError

    def set_loaders(self,loaders):
        self.loaders = loaders

    def training_step(self, batch, batch_idx):
        # forward pass
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        # calculate loss
        loss_val = self.loss(y, y_hat)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val} # Must be all Tensors
        # Log learning step as well
        # Memory Consumption possible here
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y, y_hat)

        # acc
        # labels_hat = torch.argmax(y_hat, dim=1)
        # val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        # val_acc = torch.tensor(val_acc,device=loss_val.device)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            # val_acc = val_acc.unsqueeze(0)


        return loss_val
        # output = {'val_loss': loss_val,'val_acc': val_acc}
        # can also return just a scalar instead of a dict (return loss_val)

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
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
