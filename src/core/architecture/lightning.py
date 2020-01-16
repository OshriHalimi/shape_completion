import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from util.pytorch_extensions import PytorchNet
from util.gen import banner
import pytorch_lightning as pl
from architecture.loss import F2PSMPLLoss
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from copy import deepcopy
from util.gen import time_me
import os.path as osp
import cfg


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionLightningModel(PytorchNet):
    def __init__(self, hp=()):
        super().__init__()
        self.hparams = self.add_model_specific_args(hp).parse_args()
        self.dev = 'cpu' if self.hparams.gpus is None else 'cuda'  # TODO - support with torch.device('cuda', gpu_id)
        setattr(self.hparams, 'dev', self.dev)
        exp_name = getattr(self.hparams, 'exp_name', None)
        if exp_name is not None and not exp_name:
            self.hparams.exp_name = 'default_exp'
        self.trainset, self.loss, self.loaders = None, None, None

        self._build_model()
        if self.hparams.resume_version is None:
            self._init_model()

    def hyper_params(self):
        return deepcopy(self.hparams)

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    @time_me
    def forward(self, part, template):
        raise NotImplementedError

    def init_data(self, loaders):
        # TODO - Add support for multiple Loaders for Train/Test/Validation?
        self.loaders = loaders
        faces = None
        for ldr in loaders:
            if ldr is not None:
                faces = ldr.dataset._ds_inst.faces()
                break  # This assumes validation,test and train all have the same faces.
        self.loss = F2PSMPLLoss(hparams=self.hparams, faces=faces, device=self.dev)


        # If you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.plateau_patience is not None:
            # sched = CosineAnnealingLR(optimizer, T_max=10)
            from cfg import DEF_LR_SCHED_COOLDOWN, DEF_MINIMAL_LR
            sched = ReduceLROnPlateau(opt, mode='min', patience=self.hparams.plateau_patience,
                                      cooldown=DEF_LR_SCHED_COOLDOWN, min_lr=DEF_MINIMAL_LR, verbose=True)
            # Options: factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
            # min_lr=0, eps=1e-08
            return [opt], [sched]
        else:
            return [opt]


    def training_step(self, b, _):

        pred = self.forward(b['gt_part'], b['tp'])
        loss_val = self.loss.compute(b, pred).unsqueeze(0)

        tensorboard_logs = {'train_loss': loss_val}
        return {
            'loss': loss_val,
            # 'progress_bar': tqdm_dict,
            'log': tensorboard_logs # Must be all Tensors
        }

    def validation_step(self, b, _):

        pred = self.forward(b['gt_part'], b['tp'])
        return {'val_loss': self.loss.compute(b, pred).unsqueeze(0)}

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.current_val_loss = avg_val_loss  # save current val loss state for ReduceLROnPlateau scheduler
        # TODO: Don't save! perform scheduling here and get rid of self.current_val_loss (unless appears somewhere else). Ref pytorch documentation

        tensorboard_logs = {'val_loss': avg_val_loss}
        return {"val_loss": avg_val_loss,
                "progress_bar": tensorboard_logs,
                "log": tensorboard_logs}

    def test_step(self, b, _):
        pred = self.forward(b['gt_part'], b['tp'])
        return {"test_loss": self.loss.compute(b, pred).unsqueeze(0)}

    def test_end(self, outputs):
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_test_loss}
        return {"test_loss": avg_test_loss,
                "progress_bar": tensorboard_logs,
                "log": tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return self.loaders[0]

    @pl.data_loader
    def val_dataloader(self):
        return self.loaders[1]

    @pl.data_loader
    def test_dataloader(self):
        return self.loaders[2]

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser


def train_lightning(nn, fast_dev_run=False):
    banner('Network Init')
    hp = nn.hyper_params()
    early_stop = EarlyStopping(monitor='val_loss', patience=hp.early_stop_patience, verbose=1, mode='min')
    # Consider min_delta option for EarlyStopping
    logger = TestTubeLogger(save_dir=cfg.PRIMARY_RESULTS_DIR, description=f"{hp.exp_name} Experiment",
                            name=hp.exp_name, version=hp.resume_version)
    # Support for resume_by:
    checkpoint = ModelCheckpoint(filepath=osp.join(osp.dirname(logger.experiment.log_dir), 'checkpoints'),
                                 save_top_k=0, verbose=True, prefix='weight',
                                 monitor='val_loss', mode='min', period=1)

    # NOTE: Setting logger=False can vastly improve IO bottleneck. See Issue #581
    trainer = Trainer(fast_dev_run=fast_dev_run, num_sanity_val_steps=0, weights_summary=None,
                      gpus=hp.gpus, distributed_backend=hp.distributed_backend, use_amp=hp.use_16b,
                      early_stop_callback=early_stop, checkpoint_callback=checkpoint, logger=logger,
                      min_epochs=hp.force_train_epoches)

    # More flags to consider:
    # log_gpu_memory = 'min_max' or 'all' # How to log the GPU memory
    # track_grad_norm  = 2 # Track L2 norm of the gradient # Track the Gradient Norm
    # log_save_interval = 100
    # weights_summary = 'full', 'top' , None
    # accumulate_grad_batches = 1
    banner('Training Phase')
    trainer.fit(nn)
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
