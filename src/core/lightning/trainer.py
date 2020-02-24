from lightning.pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch_lightning.loggers import TestTubeLogger
from lightning.pytorch_lightning import Trainer
from lightning.assets.completion_saver import CompletionSaver
from lightning.assets.emailer import TensorboardEmailer
from collections.abc import Sequence
from util.container import to_list, first
import lightning.assets.plotter
from util.torch.nn import TensorboardSupervisor
from util.strings import banner
from pathlib import Path
import os
import torch
import logging as log


#  This class adds additional functionality to the Lightning Trainer, wrapping it with a similar name
class LightningTrainer:
    def __init__(self, nn, loader_complex):
        # Link self to Neural Network
        self.nn = nn
        # Link Neural Network to assets:
        self.nn.assets = self
        # Link hp to self for quick access:
        self.hp = self.nn.hp

        self.data = ParametricData(loader_complex)
        self.hp = self.data.append_data_args(self.hp)

        # Training Asset place-holders:
        self.saver, self.early_stop = None, False  # Internal Trainer Assets are marked with False, not None
        # Testing Asset place-holders:
        self.plt, self.tb_sup, self.emailer = None, None,None
        # Additional Structures:
        self.trainer, self.exp_dp = None, None

    def train(self, debug_mode=False):
        banner('Training Phase')
        if not self.trainer:
            self._init_training_assets()
            log.info(f'Training on dataset: {self.data.curr_trainset_name()}')

        self._trainer(debug_mode).fit(self.nn, self.data.train_ldr, self.data.vald_ldrs, self.data.test_ldrs)
        # train_dataloader=None, val_dataloader=None, test_dataloader=None

    def test(self):
        banner('Testing Phase')
        self._trainer().test(self.nn, self.data.test_ldrs)  # Sets the trainer

    def finalize(self):
        # Called after all epochs, for cleanup
        if self.plt and self.plt.is_alive():
            self.plt.finalize()
        if self.tb_sup:
            self.tb_sup.finalize()

        # If needed, send the final report via email:
        if self.emailer:
            log.info("Sending zip with experiment specs to configured inbox")
            self.emailer.send_report(self.trainer.final_result_str)

        log.info("Cleaning up GPU memory")
        torch.cuda.empty_cache()

    def _init_training_assets(self):

        # For internal trainer:
        self.early_stop = EarlyStopping(monitor='val_loss', patience=self.hp.early_stop_patience, verbose=True,
                                        mode='min')

        if self.hp.plotter_class:
            plt_class = getattr(lightning.assets.plotter, self.hp.plotter_class)
            self.plt = plt_class(faces=self.data.faces(), n_verts=self.data.num_verts())



    def _init_trainer(self, fast_dev_run):

        # Checkpointing and Logging:
        tb_log = TestTubeLogger(save_dir=self.hp.PRIMARY_RESULTS_DIR, description=f"{self.hp.exp_name} Experiment",
                                name=self.hp.exp_name, version=self.hp.version)

        self.exp_dp = Path(os.path.dirname(tb_log.experiment.log_dir)).resolve()  # Extract experiment path
        checkpoint = ModelCheckpoint(filepath=self.exp_dp / 'checkpoints', save_top_k=1, verbose=True,
                                     prefix='weight', monitor='val_loss', mode='min', period=1)

        # Support for Auto-Tensorboard:
        if self.hp.use_auto_tensorboard > 0:
            self.tb_sup = TensorboardSupervisor(mode=self.hp.use_auto_tensorboard)

        # Support for Completion Save:
        if self.hp.save_completions > 0 and self.data.num_test_loaders() > 0:
            self.saver = CompletionSaver(exp_dir=self.exp_dp, testset_names=self.data.testset_names(),
                                         extended_save=(self.hp.save_completions == 3),
                                         f=self.data.faces() if self.hp.save_completions > 1 else None)

        if self.hp.email_report:
            self.emailer = TensorboardEmailer(exp_dp=self.exp_dp)

        self.trainer = Trainer(fast_dev_run=fast_dev_run, num_sanity_val_steps=0, weights_summary=None,
                               gpus=self.hp.gpus, distributed_backend=self.hp.distributed_backend,
                               early_stop_callback=self.early_stop, checkpoint_callback=checkpoint,
                               logger=tb_log,
                               min_epochs=self.hp.force_train_epoches,
                               max_epochs=self.hp.max_epochs,
                               print_nan_grads=False,
                               resume_cfg=self.hp.resume_cfg)
        # log_gpu_memory = 'min_max' or 'all'  # How to log the GPU memory
        # track_grad_norm = 2  # Track L2 norm of the gradient # Track the Gradient Norm
        # log_save_interval = 100
        # weights_summary = 'full', 'top', None
        # accumulate_grad_batches = 1
        log.info(f'Current run directory: {str(self.exp_dp)}')

    def _trainer(self, fast_dev_run=False):
        if not self.trainer:
            self._init_trainer(fast_dev_run)
        return self.trainer


class ParametricData:
    def __init__(self, loader_complex):
        self.train_ldr = loader_complex[0]
        self.vald_ldrs = to_list(loader_complex[1], encapsulate_none=False)
        self.test_ldrs = to_list(loader_complex[2], encapsulate_none=False)

        # Presuming all loaders stem from the SAME parametric model
        self.rep_ldr = first([self.train_ldr] + self.vald_ldrs + self.test_ldrs, lambda x: x is not None)

        # torch_faces cache
        self.torch_f = None

        self.test_set_names = []
        for i in range(self.num_test_loaders()):
            self.test_set_names.append(self.test_ldrs[i].set_name())
        self.test_set_names = tuple(self.test_set_names)
        assert len(self.test_set_names) == len(set(self.test_set_names)), "No support for non-unique sets"

        self.vald_set_names = []
        for i in range(self.num_vald_loaders()):
            self.vald_set_names.append(self.vald_ldrs[i].set_name())
        self.vald_set_names = tuple(self.vald_set_names)
        assert len(self.vald_set_names) == len(set(self.vald_set_names)), "No support for non-unique sets"

    def testset_names(self):
        return self.test_set_names

    def valdset_names(self):
        return self.vald_set_names

    def curr_trainset_name(self):
        if self.num_train_loaders() == 0:
            return None
        else:
            return self.train_ldr.set_name()

    def num_train_loaders(self):
        return 1 if self.train_ldr else 0

    def num_vald_loaders(self):
        return len(self.vald_ldrs)

    def num_test_loaders(self):
        return len(self.test_ldrs)

    def id2vald_ds(self, set_id):
        return self.vald_set_names[set_id]

    def id2test_ds(self, set_id):
        return self.test_set_names[set_id]

    def faces(self):
        return self.rep_ldr.faces()

    def torch_faces(self):
        assert self.torch_f
        return self.torch_f

    def num_verts(self):
        return self.rep_ldr.num_verts()

    def num_faces(self):
        return self.rep_ldr.num_faces()

    def append_data_args(self, hp):

        if self.train_ldr:
            setattr(hp, f'train_ds', self.train_ldr.recon_table())
        for i in range(self.num_vald_loaders()):
            setattr(hp, f'vald_ds_{i}', self.vald_ldrs[i].recon_table())
        for i in range(self.num_test_loaders()):
            setattr(hp, f'test_ds_{i}', self.test_ldrs[i].recon_table())

        setattr(hp, 'compute_output_normals', hp.VIS_SHOW_NORMALS or
                hp.lambdas[1] > 0 or hp.lambdas[4] > 0 or hp.lambdas[5] > 0)

        if hp.compute_output_normals:
            assert hp.in_channels >= 6, "In channels not aligned to loss/plot config"
            self.torch_f = torch.from_numpy(self.faces()).long().to(device=hp.dev, non_blocking=hp.NON_BLOCKING)

        return hp
