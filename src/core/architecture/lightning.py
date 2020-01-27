import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # , CosineAnnealingLR

import pytorch_lightning as pl
from architecture.loss import F2PSMPLLoss
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from util.torch_ext import PytorchNet
from util.gen import banner, get_book_variable_module_name, time_me
from util.container import first
from util.mesh_io import write_obj
from copy import deepcopy
from pathlib import Path
import os.path as osp


# ----------------------------------------------------------------------------------------------------------------------
#                                                         Stand alone
# ----------------------------------------------------------------------------------------------------------------------
def extend_hyper_params(hp, hp_data_tables):
    # Note: For data compatibility, hp.dev is extended in the constructor
    # Dataset:
    for k, v in hp_data_tables.items():
        setattr(hp, k, v)

    # Config Variables
    for k, v in get_book_variable_module_name('cfg').items():  # Only import non-class/module types
        setattr(hp, k, v)

    # Experiment:
    if hp.exp_name is None or not hp.exp_name:
        hp.exp_name = 'default_exp'

    return hp


def train_lightning(nn, fast_dev_run=False):
    banner('Network Init')
    nn.identify_system()
    hp = nn.hyper_params()

    # NOTE: Setting logger=False may vastly improve IO bottleneck. See Issue #581
    trainer = Trainer(fast_dev_run=fast_dev_run, num_sanity_val_steps=0, weights_summary=None,
                      gpus=hp.gpus, distributed_backend=hp.distributed_backend, use_amp=hp.use_16b,
                      early_stop_callback=nn.early_stop, checkpoint_callback=nn.checkpoint, logger=nn.tb_logger,
                      min_epochs=hp.force_train_epoches, report_loss_per_batch=hp.REPORT_LOSS_PER_BATCH,
                      max_epochs=hp.MAX_EPOCHS)

    """ More flags to consider:
    log_gpu_memory = 'min_max' or 'all' # How to log the GPU memory
    track_grad_norm  = 2 # Track L2 norm of the gradient # Track the Gradient Norm
    log_save_interval = 100
    weights_summary = 'full', 'top' , None
    accumulate_grad_batches = 1
    """
    # banner('Training Phase')
    # trainer.fit(nn)
    banner('Testing Phase')
    trainer.test(nn)


def test_lightning(nn):
    pass
    # nn = F2PEncoderDecoder.load_from_metrics(
    #     weights_path='/path/to/pytorch_checkpoint.ckpt',
    #     tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
    #     on_gpu=True,
    #     map_location=None
    # )
    # pretrained_model = MyLightningModule.load_from_checkpoint(
    #     checkpoint_path='/path/to/pytorch_checkpoint.ckpt'
    # )


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionLightningModel(PytorchNet):
    def __init__(self, hp=()):
        super().__init__()
        self.hparams = self.add_model_specific_args(hp).parse_args()
        dev = 'cpu' if self.hparams.gpus is None else torch.device('cuda', torch.cuda.current_device())
        setattr(self.hparams, 'dev', dev)  # Not very smart to place it outside the extend_function

        # Book-keeping:
        self.opt, self.loss, self.loaders = None, None, None
        self.early_stop, self.checkpoint, self.tb_logger = None, None, None
        self.completions_dp, self.exp_dp, self.f = None, None, None

        self._build_model()  # Must be done after the setting of hparams
        if hp and self.hparams.resume_version is None:
            self._init_model()

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def forward(self, part, template):
        raise NotImplementedError

    def init_data(self, loaders):
        self.loaders = loaders
        hp_data_tables = {}
        for assignment, ldr in zip(('train_ds', 'vald_ds', 'test_ds'), loaders):
            hp_data_tables[assignment] = None if ldr is None else ldr.recon_table()

        self.hparams = extend_hyper_params(self.hparams, hp_data_tables)
        self._init_trainer_collaterals()

    def configure_optimizers(self):

        f = first(self.loaders, lambda x: x is not None).dataset._ds_inst.faces()
        self.loss = F2PSMPLLoss(hp=self.hparams, faces=f)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.plateau_patience is not None:
            # sched = CosineAnnealingLR(optimizer, T_max=10)
            sched = ReduceLROnPlateau(self.opt, mode='min', patience=self.hparams.plateau_patience, verbose=True,
                                      cooldown=self.hparams.DEF_LR_SCHED_COOLDOWN, eps=self.hparams.DEF_MINIMAL_LR)
            # Options: factor=0.1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
            return [self.opt], [sched]
        else:
            return [self.opt]

    def training_step(self, b, _):
        pred = self.forward(b['gt_part'], b['tp'])
        loss = self.loss.compute(b, pred).unsqueeze(0)
        return {
            'loss': loss,  # Must use 'loss' instead of 'train_loss' due to lightning framework
            'log': {'loss': loss}
        }

    def validation_step(self, b, _):

        pred = self.forward(b['gt_part'], b['tp'])
        return {'val_loss': self.loss.compute(b, pred).unsqueeze(0)}

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        lr = self.learning_rate(self.opt)

        logs = {'val_loss': avg_val_loss, 'lr': lr}
        return {"val_loss": avg_val_loss,
                "progress_bar": logs,
                "log": logs}

    def test_step(self, b, _):

        gtrb = self.forward(b['gt_part'], b['tp'])
        if self.hparams.save_completions > 0:
            self.save_completions_by_batch(gtrb, b['gt_hi'], b['tp_hi'])

        return {"test_loss": self.loss.compute(b, gtrb).unsqueeze(0)}

    def test_end(self, outputs):
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_test_loss}
        return {"test_loss": avg_test_loss,
                "progress_bar": logs,
                "log": logs}

    def save_completions_by_batch(self, gtrb, gt_hi_b, tp_hi_b):
        gtrb = gtrb.cpu().numpy()
        for i, (gt_hi, tp_hi) in enumerate(zip(gt_hi_b, tp_hi_b)):
            gt_hi = '_'.join(str(x) for x in gt_hi)
            tp_hi = '_'.join(str(x) for x in tp_hi)  # TODO - Add support for P2P
            gtr_v = gtrb[i, :, :3]
            fp = self.completions_dp / f'{self.test_ds_name}_gthi_{gt_hi}_tphi_{tp_hi}_res.obj'
            write_obj(fp, gtr_v, self.f)

    def hyper_params(self):
        return deepcopy(self.hparams)

    def _init_trainer_collaterals(self):

        hp = self.hparams
        # TODO - decide when to use these, and when not to
        self.early_stop = EarlyStopping(monitor='val_loss', patience=hp.early_stop_patience, verbose=1, mode='min')
        self.tb_logger = TestTubeLogger(save_dir=hp.PRIMARY_RESULTS_DIR, description=f"{hp.exp_name} Experiment",
                                        name=hp.exp_name, version=hp.resume_version)
        self.exp_dp = Path(osp.dirname(self.tb_logger.experiment.log_dir)).resolve()

        # Support for completions:
        if hp.save_completions > 0:
            self.test_ds_name = self.hparams.test_ds['dataset_name']
            self.completions_dp = self.exp_dp / f'{self.test_ds_name}_completions'
            self.completions_dp.mkdir(parents=True, exist_ok=True)
            if hp.save_completions == 2:  # With faces
                self.f = first(self.loaders, lambda x: x is not None).dataset._ds_inst.faces()

        self.checkpoint = ModelCheckpoint(filepath=self.exp_dp / 'checkpoints', save_top_k=0,
                                          verbose=True, prefix='weight', monitor='val_loss', mode='min', period=1)

        # self.example_input_array = torch.rand(5, 28 * 28) # Used by weight summary to show in/out for each layer

    @pl.data_loader
    def train_dataloader(self):
        return self.loaders[0]

    @pl.data_loader
    def val_dataloader(self):
        return self.loaders[1]

    @pl.data_loader
    def test_dataloader(self):
        return self.loaders[2]

    @staticmethod  # You need to override this method
    def add_model_specific_args(parent_parser):
        return parent_parser
