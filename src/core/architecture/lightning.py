import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from util.pytorch_extensions import PytorchNet
import pytorch_lightning as pl
from architecture.loss import F2PSMPLLoss
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import cfg
import os


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionLightningModel(PytorchNet):
    def __init__(self, hp=()):
        super().__init__()
        self.hparams = self.add_model_specific_args(hp).parse_args()
        self.dev = 'cpu' if self.hparams.gpus is None else 'cuda'  # TODO - Insert distributed support
        setattr(self.hparams, 'dev', self.dev)
        self.trainset, self.loss, self.loaders = None, None, None

        self._build_model()
        if self.hparams.resume_by:
            pass  # TODO
        else:
            self._init_model()

    def hyper_params(self):
        return self.hparams

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def forward(self, part, template):
        raise NotImplementedError

    def init_data(self, loaders, faces):
        # TODO - loaders as dict ?
        dev = 'cpu' if self.hparams.gpus is None else 'cuda'

        self.loaders = loaders
        # This assumes validation,test and train all have the same faces.
        self.loss = F2PSMPLLoss(hparams=self.hparams, faces=faces, device=dev)
        # If you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = CosineAnnealingLR(optimizer, T_max=10)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.hparams.plateau_patience, verbose=False)
        # Options: factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
        # eps=1e-08
        return [optimizer], [scheduler]

    def training_step(self, b, _):

        pred = self.forward(b['gt_part_v'], b['tp_v'])
        loss_val = self.loss.compute(b, pred).unsqueeze(0)
        return {
            'loss': loss_val,
            # 'progress_bar': tqdm_dict,
            'log': {'train_loss': loss_val}  # Must be all Tensors
        }

    def validation_step(self, b, _):

        # TODO - What more do we need to log? Learning Step? Memory Consumption?
        pred = self.forward(b['gt_part_v'], b['tp_v'])
        loss_val = self.loss.compute(b, pred).unsqueeze(0)
        return {'val_loss': loss_val}

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        avg_val_loss = 0.0
        for output in outputs:
            avg_val_loss += output["val_loss"].mean() / len(outputs)

        logs = {'avg_val_loss': avg_val_loss}
        return {"avg_val_loss": avg_val_loss,
                "progress_bar": logs,
                "log": logs}

    def test_step(self, b, _):
        pred = self.forward(b['gt_part_v'], b['tp_v'])
        loss_val = self.loss.compute(b, pred).unsqueeze(0)
        return {"test_loss": loss_val}

    def test_end(self, outputs):
        avg_test_loss = 0.0
        for output in outputs:
            avg_test_loss += output["test_loss"].mean() / len(outputs)

        logs = {'avg_test_loss': avg_test_loss}
        return {"avg_test_loss": avg_test_loss,
                "progress_bar": logs,
                "log": logs}

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
    hp = nn.hyper_params()

    if not hp.exp_name or hp.exp_name is None:
        hp.exp_name = 'default_exp'
    tgt_dir = cfg.PRIMARY_RESULTS_DIR / hp.exp_name
    tgt_dir /= f'version_{get_exp_version(tgt_dir)}'
    tgt_dir = tgt_dir.resolve()
    early_stop = EarlyStopping(monitor='avg_val_loss', patience=hp.early_stop_patience, verbose=True, mode='min')
    # Consider min_delta option for EarlyStopping
    logger = TestTubeLogger(save_dir=tgt_dir, description="BluBluBlu")
    # log.rank = 0
    checkpoint = ModelCheckpoint(
        filepath=tgt_dir / 'checkpoints',  # / 'weights_epoch_{epoch:02d}_vloss_{avg_val_loss:.5f}.ckpt',
        save_best_only=True, verbose=True,prefix='weight',
        monitor='avg_val_loss', mode='min', period=1)

    trainer = Trainer(nb_sanity_val_steps=1, gpus=hp.gpus, distributed_backend=hp.distributed_backend,
                      use_amp=hp.use_16b, fast_dev_run=fast_dev_run, default_save_path=cfg.PRIMARY_RESULTS_DIR,
                      early_stop_callback=early_stop, checkpoint_callback=checkpoint, max_nb_epochs=hp.n_epoch,
                      logger=logger)

    # More flags to consider:
    # log_gpu_memory = 'min_max' or 'all' # How to log the GPU memory
    # track_grad_norm  = 2 # Track L2 norm of the gradient # Track the Gradient Norm
    # log_save_interval = 100
    # weights_summary = 'full', 'top' , None
    # accumulate_grad_batches = 1

    trainer.fit(nn)
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


def get_exp_version(cache_dir):
    last_version = -1
    try:
        for f in os.listdir(cache_dir):
            if 'version_' in f:
                file_parts = f.split('_')
                version = int(file_parts[-1])
                last_version = max(last_version, version)
    except:  # No such dir
        pass

    return last_version + 1
