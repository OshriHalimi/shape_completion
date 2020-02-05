import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # , CosineAnnealingLR
import pytorch_lightning as pl
from architecture.loss import F2PSMPLLoss
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import mesh.io
from mesh.plot import ParallelPlotter
from util.torch_nn import PytorchNet
from util.string_op import banner
from util.func import all_variables_by_module_name
from util.container import first
from copy import deepcopy
from pathlib import Path
import os.path as osp
from util.torch_nn import TensorboardSupervisor
import logging
from functools import reduce


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Trainer
# ----------------------------------------------------------------------------------------------------------------------

def lightning_trainer(nn, fast_dev_run=False):
    hp = nn.hyper_params()
    # TODO - Add support for logger = False
    trainer = Trainer(fast_dev_run=fast_dev_run, num_sanity_val_steps=0, weights_summary=None,
                      gpus=hp.gpus, distributed_backend=hp.distributed_backend, use_amp=hp.use_16b,
                      early_stop_callback=nn.early_stop, checkpoint_callback=nn.checkpoint, logger=nn.tb_logger,
                      min_epochs=hp.force_train_epoches, report_loss_per_batch=hp.REPORT_LOSS_PER_BATCH,
                      max_epochs=hp.MAX_EPOCHS,print_nan_grads=False)
    """ More flags to consider:
    log_gpu_memory = 'min_max' or 'all' # How to log the GPU memory
    track_grad_norm  = 2 # Track L2 norm of the gradient # Track the Gradient Norm
    log_save_interval = 100
    weights_summary = 'full', 'top' , None
    accumulate_grad_batches = 1
    """
    return trainer


def test_lightning(nn):
    # TODO - Complete this
    print(nn)
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
        self.hparams = append_config_args(self.add_model_specific_args(hp).parse_args())

        # Book-keeping:
        self.opt, self.loss, self.early_stop, self.checkpoint, self.tb_logger = None, None, None, None, None
        self.loaders, self.completions_dp, self.exp_dp, self.f, self.n_f, self.n_v = None, None, None, None, None, None
        self.save_func, self.tb_sub = None, None

        self._build_model()  # Set hparams before this
        if self.hparams.do_weight_init:
            # Set network weight precision
            self.type(dst_type=getattr(torch, self.hparams.UNIVERSAL_PRECISION))  # TODO - Might be problematic
            self._init_model()

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def forward(self, part, template):
        raise NotImplementedError

    def init_data(self, loaders):

        # Assign Loaders:
        self.loaders = loaders

        # Assign faces & Number of vertices - TODO - Remember this strong assumption
        ldr = first(self.loaders, lambda x: x is not None)
        self.f = ldr.dataset._ds_inst.faces()
        self.n_f = self.f.shape[0]
        self.n_v = ldr.dataset._ds_inst.shape()[0]

        # Extend Hyper-Parameters:
        self.hparams = append_data_args(self.hparams, loaders)

        self._init_trainer_collaterals()

    def _init_trainer_collaterals(self):

        hp = self.hparams

        # TODO - Is all of this needed when only testing? What about other cases?
        self.early_stop = EarlyStopping(monitor='val_loss', patience=hp.early_stop_patience, verbose=1, mode='min')
        self.tb_logger = TestTubeLogger(save_dir=hp.PRIMARY_RESULTS_DIR, description=f"{hp.exp_name} Experiment",
                                        name=hp.exp_name, version=hp.resume_version)
        self.exp_dp = Path(osp.dirname(self.tb_logger.experiment.log_dir)).resolve()
        self.checkpoint = ModelCheckpoint(filepath=self.exp_dp / 'checkpoints', save_top_k=0,
                                          verbose=True, prefix='weight', monitor='val_loss', mode='min', period=1)

        logging.info(f'Current run directory: {str(self.exp_dp)}')
        if hp.use_tensorboard:
            self.tb_sub = TensorboardSupervisor()

        # Support for completions:
        if hp.save_completions > 0:
            self.test_ds_name = self.hparams.test_ds['dataset_name']
            self.completions_dp = self.exp_dp / f'{self.test_ds_name}_completions'
            self.completions_dp.mkdir(parents=True, exist_ok=True)
            self.save_func = getattr(mesh.io, f'write_{hp.SAVE_MESH_AS}')

        # Support for parallel plotter:
        if hp.use_parallel_plotter:
            # logging.info('Initializing Parallel Plotter')
            self.plt = ParallelPlotter(faces=self.f, n_verts=self.n_v)

    def configure_optimizers(self):
        self.loss = F2PSMPLLoss(hp=self.hparams, f=self.f)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        if self.hparams.plateau_patience is not None:
            # sched = CosineAnnealingLR(optimizer, T_max=10)
            sched = ReduceLROnPlateau(self.opt, mode='min', patience=self.hparams.plateau_patience, verbose=True,
                                      cooldown=self.hparams.DEF_LR_SCHED_COOLDOWN, eps=self.hparams.DEF_MINIMAL_LR)
            # Options: factor=0.1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
            return [self.opt], [sched]
        else:
            return [self.opt]

    def training_step(self, b, batch_idx):
        pred = self.forward(b['gt_part'], b['tp'])
        loss_dict = self.loss.compute(b, pred[0]) # TODO:loss and pred can vary from network to network
        loss_dict = {f'{k}_train': v for k, v in loss_dict.items()}  # make different logs for train, test, validation
        train_loss = loss_dict['total_loss_train']
        logs = loss_dict

        if self.hparams.use_parallel_plotter and batch_idx == 0:  # On first batch
            self.plt.cache(self._prepare_plotter_dict(b, pred))  # New tensors, without grad

        return {
            'loss': train_loss,  # Must use 'loss' instead of 'train_loss' due to lightning framework
            'log': logs
        }

    def on_train_end(self):  # TODO - Move this to somewhere smarter
        # Called after all epochs, for cleanup
        if self.hparams.use_parallel_plotter and self.plt.is_alive():
            self.plt.finalize()
        if self.hparams.use_tensorboard:
            self.tb_sub.finalize()

    def validation_step(self, b, batch_idx):
        pred = self.forward(b['gt_part'], b['tp'])

        if self.hparams.use_parallel_plotter and batch_idx == 0:  # On first batch
            new_data = (self.plt.uncache(), self._prepare_plotter_dict(b, pred))
            self.plt.push(new_data=new_data, new_epoch=self.current_epoch)

        loss_dict = self.loss.compute(b, pred[0])  # TODO:loss and pred can vary from network to network
        return loss_dict

    def validation_end(self, outputs):
        # average the values with same keys
        loss_dict = {}
        keys = outputs[0].keys()
        for k in keys:
            loss_dict[k] = torch.stack([x[k] for x in outputs]).mean()
        loss_dict = {f'{k}_val': v for k, v in loss_dict.items()} # make different logs for train, test, validation
        val_loss_avg = loss_dict['total_loss_val']

        lr = self.learning_rate(self.opt)  # Also log learning rate
        pb = {'val_loss': val_loss_avg, 'lr': lr}
        logs = loss_dict
        logs.update(lr=lr)

        # This must be kept as "val_loss" and not "avg_val_loss" due to lightning bug
        return {"val_loss": val_loss_avg,
                "progress_bar": pb,
                "log": logs}

    def test_step(self, b, _):

        pred = self.forward(b['gt_part'], b['tp'])
        if self.hparams.save_completions > 0:
            self._save_completions_by_batch(pred[0], b['gt_hi'], b['tp_hi'])  # TODO:pred can vary from network to network

        loss_dict = self.loss.compute(b, pred[0])  # TODO:loss and pred can vary from network to network
        return loss_dict

    def test_end(self, outputs):
        loss_dict = {}
        keys = outputs[0].keys()
        for k in keys:
            loss_dict[k] = torch.stack([x[k] for x in outputs]).mean()
        loss_dict = {f'{k}_test': v for k, v in loss_dict.items()} # make different logs for train, test, validation
        avg_test_loss = loss_dict['total_loss_test']

        pb = {'test_loss': avg_test_loss}
        logs = loss_dict

        return {"test_loss": avg_test_loss,
                "progress_bar": pb,
                "log": logs}

    def _prepare_plotter_dict(self, b, gtrb):

        # TODO - Support normals
        max_b_idx = self.hparams.VIS_N_MESH_SETS
        return {'gt': b['gt'].detach().cpu().numpy()[:max_b_idx, :, :3],
                'tp': b['tp'].detach().cpu().numpy()[:max_b_idx, :, :3],
                'gtrb': gtrb.detach().cpu().numpy()[:max_b_idx],
                'gt_hi': b['gt_hi'][:max_b_idx],
                'tp_hi': b['tp_hi'][:max_b_idx],
                'gt_mask_vi': b['gt_mask_vi'][:max_b_idx]}

    def _save_completions_by_batch(self, gtrb, gt_hi_b, tp_hi_b):
        gtrb = gtrb.cpu().numpy()
        for i, (gt_hi, tp_hi) in enumerate(zip(gt_hi_b, tp_hi_b)):
            gt_hi = '_'.join(str(x) for x in gt_hi)
            tp_hi = '_'.join(str(x) for x in tp_hi)  # TODO - Add support for P2P
            gtr_v = gtrb[i, :, :3]
            fp = self.completions_dp / f'{self.test_ds_name}_gthi_{gt_hi}_tphi_{tp_hi}_res'
            self.save_func(fp, gtr_v, self.f)

    def hyper_params(self):
        return deepcopy(self.hparams)

    @staticmethod  # You need to override this method
    def add_model_specific_args(parent_parser):
        return parent_parser

    @pl.data_loader
    def train_dataloader(self):
        return self.loaders[0]

    @pl.data_loader
    def val_dataloader(self):
        return self.loaders[1]

    @pl.data_loader
    def test_dataloader(self):
        return self.loaders[2]


# ----------------------------------------------------------------------------------------------------------------------
#                                        Hyper Parameter Extents
# ----------------------------------------------------------------------------------------------------------------------
def append_config_args(hp):
    # Config Variables
    for k, v in all_variables_by_module_name('cfg').items():  # Only import non-class/module types
        setattr(hp, k, v)

    if hasattr(hp, 'gpus'):  # This is here to allow init of network with only model params (no argin)

        # Device - TODO - Does this support multiple GPU ?
        dev = torch.device('cpu') if hp.gpus is None else torch.device('cuda', torch.cuda.current_device())
        setattr(hp, 'dev', dev)

        # Experiment:
        if hp.exp_name is None or not hp.exp_name:
            hp.exp_name = 'default_exp'

        # Weight Init Flag
        setattr(hp, 'do_weight_init', hp.resume_version is None)

        # Correctness of config parameters:
        assert hp.VIS_N_MESH_SETS <= hp.batch_size, \
            f"Plotter needs requires batch size >= N_MESH_SETS={hp.VIS_N_MESH_SETS}"

    else:
        setattr(hp, 'do_weight_init', True)

    return hp


def append_data_args(hp, loaders):
    for set_name, ldr in zip(('train_ds', 'vald_ds', 'test_ds'), loaders):
        table = None if ldr is None else ldr.recon_table()
        setattr(hp, set_name, table)

    # TODO - This might be a wrong assumption
    setattr(hp, 'use_parallel_plotter', hp.use_parallel_plotter and loaders[0] is not None)

    return hp
