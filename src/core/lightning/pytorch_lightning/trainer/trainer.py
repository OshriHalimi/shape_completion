import os
import sys
import warnings
import logging as log

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from torch.optim.optimizer import Optimizer

from lightning.pytorch_lightning.trainer.auto_mix_precision import TrainerAMPMixin
from lightning.pytorch_lightning.trainer.callback_config import TrainerCallbackConfigMixin
from lightning.pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from lightning.pytorch_lightning.trainer.distrib_data_parallel import TrainerDDPMixin
from lightning.pytorch_lightning.trainer.distrib_parts import (
    TrainerDPMixin,
    parse_gpu_ids,
    determine_root_gpu_device
)
from lightning.pytorch_lightning.core.lightning import LightningModule
from lightning.pytorch_lightning.trainer.evaluation_loop import TrainerEvaluationLoopMixin
from lightning.pytorch_lightning.trainer.logging import TrainerLoggingMixin
from lightning.pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from lightning.pytorch_lightning.trainer.training_io import TrainerIOMixin
from lightning.pytorch_lightning.trainer.training_loop import TrainerTrainLoopMixin
from lightning.pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from lightning.pytorch_lightning.utilities.debugging import MisconfigurationException
from lightning.pytorch_lightning.profiler import Profiler, PassThroughProfiler

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


class Trainer(TrainerIOMixin,
              TrainerDPMixin,
              TrainerDDPMixin,
              TrainerLoggingMixin,
              TrainerModelHooksMixin,
              TrainerTrainingTricksMixin,
              TrainerDataLoadingMixin,
              TrainerAMPMixin,
              TrainerEvaluationLoopMixin,
              TrainerTrainLoopMixin,
              TrainerCallbackConfigMixin,
              ):

    def __init__(
            self,
            logger=True,
            checkpoint_callback=True,
            early_stop_callback=None,
            default_save_path=None,
            gradient_clip_val=0,
            gradient_clip=None,  # backward compatible, todo: remove in v0.8.0
            process_position=0,
            nb_gpu_nodes=None,  # backward compatible, todo: remove in v0.8.0
            num_nodes=1,
            gpus=None,
            num_tpu_cores=None,
            log_gpu_memory=None,
            show_progress_bar=True,
            overfit_pct=0.0,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            accumulate_grad_batches=1,
            max_nb_epochs=None,  # backward compatible, todo: remove in v0.8.0
            min_nb_epochs=None,  # backward compatible, todo: remove in v0.8.0
            max_epochs=1000,
            min_epochs=1,
            max_steps=None,
            min_steps=None,
            train_percent_check=1.0,
            val_percent_check=1.0,
            test_percent_check=1.0,
            val_check_interval=1.0,
            log_save_interval=100,
            row_log_interval=10,
            add_row_log_interval=None,  # backward compatible, todo: remove in v0.8.0
            distributed_backend=None,
            use_amp=False,  # backward compatible, todo: remove in v0.8.0
            precision=32,
            print_nan_grads=False,
            weights_summary='full',
            weights_save_path=None,
            amp_level='O1',
            nb_sanity_val_steps=None,  # backward compatible, todo: remove in v0.8.0
            num_sanity_val_steps=5,
            truncated_bptt_steps=None,
            resume_from_checkpoint=None,
            profiler=None,
            resume_cfg=(True, True)  # MANO ADDED
    ):
        r"""

        Customize every aspect of training via flags

        Args:
            logger (:class:`.Logger`): Logger for experiment tracking.
                Example::

                    from pytorch_lightning.loggers import TensorBoardLogger

                    # default logger used by trainer
                    logger = TensorBoardLogger(
                        save_dir=os.getcwd(),
                        version=self.slurm_job_id,
                        name='lightning_logs'
                    )

                    Trainer(logger=logger)

            checkpoint_callback (:class:`CheckpointCallback`): Callback for checkpointing.
                Example::

                    from pytorch_lightning.callbacks import ModelCheckpoint

                    # default used by the Trainer
                    checkpoint_callback = ModelCheckpoint(
                        filepath=os.getcwd(),
                        save_best_only=True,
                        verbose=True,
                        monitor='val_loss',
                        mode='min',
                        prefix=''
                    )

                    trainer = Trainer(checkpoint_callback=checkpoint_callback)

            early_stop_callback (:class:`.EarlyStopping`): Callback for early stopping. If
                set to ``True``, then the default callback monitoring ``'val_loss'`` is created.
                Will raise an error if ``'val_loss'`` is not found.
                If set to ``False``, then early stopping will be disabled.
                If set to ``None``, then the default callback monitoring ``'val_loss'`` is created.
                If ``'val_loss'`` is not found will work as if early stopping is disabled.
                Default: ``None``.
                Example::

                    from pytorch_lightning.callbacks import EarlyStopping

                    # default used by the Trainer
                    early_stop_callback = EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                        strict=False,
                        verbose=False,
                        mode='min'
                    )

                    trainer = Trainer(early_stop_callback=early_stop_callback)

            default_save_path (str): Default path for logs and weights when no logger/ckpt_callback passed
                Example::

                    # default used by the Trainer
                    trainer = Trainer(default_save_path=os.getcwd())

            gradient_clip_val (float): 0 means don't clip.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(gradient_clip_val=0.0)

            gradient_clip (int):
                .. warning: .. deprecated:: 0.5.0
                    Use `gradient_clip_val` instead. Will remove 0.8.0.

            process_position (int): orders the tqdm bar when running multiple models on same machine.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(process_position=0)

            num_nodes (int): number of GPU nodes for distributed training.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(num_nodes=1)

                    # to train on 8 nodes
                    trainer = Trainer(num_nodes=8)

            nb_gpu_nodes (int):
                ..warning:: .. deprecated:: 0.5.0
                    Use `num_nodes` instead. Will remove 0.8.0.

            gpus (list|str|int): Which GPUs to train on.
                Example::

                    # default used by the Trainer (ie: train on CPU)
                    trainer = Trainer(gpus=None)

                    # int: train on 2 gpus
                    trainer = Trainer(gpus=2)

                    # list: train on GPUs 1, 4 (by bus ordering)
                    trainer = Trainer(gpus=[1, 4])
                    trainer = Trainer(gpus='1, 4') # equivalent

                    # -1: train on all gpus
                    trainer = Trainer(gpus=-1)
                    trainer = Trainer(gpus='-1') # equivalent

                    # combine with num_nodes to train on multiple GPUs across nodes
                    trainer = Trainer(gpus=2, num_nodes=4) # uses 8 gpus in total

            num_tpu_cores (int): How many TPU cores to train on (1 or 8).
                A single TPU v2 or v3 has 8 cores. A TPU pod has
                up to 2048 cores. A slice of a POD means you get as many cores
                as you request.

                You MUST use DistributedDataSampler with your dataloader for this
                to work. Your effective batch size is batch_size * total tpu cores.

                This parameter can be either 1 or 8.

                Example::

                    # your_trainer_file.py

                    # default used by the Trainer (ie: train on CPU)
                    trainer = Trainer(num_tpu_cores=None)

                    # int: train on a single core
                    trainer = Trainer(num_tpu_cores=1)

                    # int: train on all cores few cores
                    trainer = Trainer(num_tpu_cores=8)

                    # for 8+ cores must submit via xla script with
                    # a max of 8 cores specified. The XLA script
                    # will duplicate script onto each TPU in the POD
                    trainer = Trainer(num_tpu_cores=8)

                    # -1: train on all available TPUs
                    trainer = Trainer(num_tpu_cores=-1)

            To train on more than 8 cores (ie: a POD),
            submit this script using the xla_dist script.

            Example::

                $ python -m torch_xla.distributed.xla_dist
                --tpu=$TPU_POD_NAME
                --conda-env=torch-xla-nightly
                --env=XLA_USE_BF16=1
                -- python your_trainer_file.py

            log_gpu_memory (str): None, 'min_max', 'all'. Might slow performance
                because it uses the output of nvidia-smi.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(log_gpu_memory=None)

                    # log all the GPUs (on master node only)
                    trainer = Trainer(log_gpu_memory='all')

                    # log only the min and max memory on the master node
                    trainer = Trainer(log_gpu_memory='min_max')

            show_progress_bar (bool): If true shows tqdm progress bar
                Example::

                    # default used by the Trainer
                    trainer = Trainer(show_progress_bar=True)

            overfit_pct (float): uses this much data of all datasets.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(overfit_pct=0.0)

                    # use only 1% of the train, test, val datasets
                    trainer = Trainer(overfit_pct=0.01)

            track_grad_norm (int): -1 no tracking. Otherwise tracks that norm
                Example::

                    # default used by the Trainer
                    trainer = Trainer(track_grad_norm=-1)

                    # track the 2-norm
                    trainer = Trainer(track_grad_norm=2)

            check_val_every_n_epoch (int): Check val every n train epochs.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(check_val_every_n_epoch=1)

                    # run val loop every 10 training epochs
                    trainer = Trainer(check_val_every_n_epoch=10)

            fast_dev_run (bool): runs 1 batch of train, test  and val to find any bugs (ie: a sort of unit test).
                Example::

                    # default used by the Trainer
                    trainer = Trainer(fast_dev_run=False)

                    # runs 1 train, val, test  batch and program ends
                    trainer = Trainer(fast_dev_run=True)

            accumulate_grad_batches (int|dict): Accumulates grads every k batches or as set up in the dict.
                Example::

                    # default used by the Trainer (no accumulation)
                    trainer = Trainer(accumulate_grad_batches=1)

                    # accumulate every 4 batches (effective batch size is batch*4)
                    trainer = Trainer(accumulate_grad_batches=4)

                    # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
                    trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})

            max_epochs (int): Stop training once this number of epochs is reached.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(max_epochs=1000)

            max_nb_epochs (int):
                .. warning:: .. deprecated:: 0.5.0
                    Use `max_epochs` instead. Will remove 0.8.0.

            min_epochs (int): Force training for at least these many epochs
                Example::

                    # default used by the Trainer
                    trainer = Trainer(min_epochs=1)

            min_nb_epochs (int):
                .. warning:: .. deprecated:: 0.5.0
                    Use `min_nb_epochs` instead. Will remove 0.8.0.

            max_steps (int): Stop training after this number of steps. Disabled by default (None).
                Training will stop if max_steps or max_epochs have reached (earliest).
                Example::

                    # Stop after 100 steps
                    trainer = Trainer(max_steps=100)

            min_steps(int): Force training for at least these number of steps. Disabled by default (None).
                Trainer will train model for at least min_steps or min_epochs (latest).
                Example::

                    # Run at least for 100 steps (disable min_epochs)
                    trainer = Trainer(min_steps=100, min_epochs=0)

            train_percent_check (int): How much of training dataset to check.
                Useful when debugging or testing something that happens at the end of an epoch.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(train_percent_check=1.0)

                    # run through only 25% of the training set each epoch
                    trainer = Trainer(train_percent_check=0.25)

            val_percent_check (int): How much of validation dataset to check.
                Useful when debugging or testing something that happens at the end of an epoch.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(val_percent_check=1.0)

                    # run through only 25% of the validation set each epoch
                    trainer = Trainer(val_percent_check=0.25)

            test_percent_check (int): How much of test dataset to check.
                Useful when debugging or testing something that happens at the end of an epoch.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(test_percent_check=1.0)

                    # run through only 25% of the test set each epoch
                    trainer = Trainer(test_percent_check=0.25)

            val_check_interval (float|int): How often within one training epoch to check the validation set
                If float, % of tng epoch. If int, check every n batch
                Example::

                    # default used by the Trainer
                    trainer = Trainer(val_check_interval=1.0)

                    # check validation set 4 times during a training epoch
                    trainer = Trainer(val_check_interval=0.25)

                    # check validation set every 1000 training batches
                    # use this when using iterableDataset and your dataset has no length
                    # (ie: production cases with streaming data)
                    trainer = Trainer(val_check_interval=1000)

            log_save_interval (int): Writes logs to disk this often
                Example::

                    # default used by the Trainer
                    trainer = Trainer(log_save_interval=100)

            row_log_interval (int): How often to add logging rows (does not write to disk)
                Example::

                    # default used by the Trainer
                    trainer = Trainer(row_log_interval=10)

            add_row_log_interval (int):
                .. warning:: .. deprecated:: 0.5.0
                    Use `row_log_interval` instead. Will remove 0.8.0.

            distributed_backend (str): The distributed backend to use.
                Options: 'dp', 'ddp', 'ddp2'.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(distributed_backend=None)

                    # dp = DataParallel (split a batch onto k gpus on same machine).
                    trainer = Trainer(gpus=2, distributed_backend='dp')

                    # ddp = DistributedDataParallel
                    # Each gpu trains by itself on a subset of the data.
                    # Gradients sync across all gpus and all machines.
                    trainer = Trainer(gpus=2, num_nodes=2, distributed_backend='ddp')

                    # ddp2 = DistributedDataParallel + dp
                    # behaves like dp on every node
                    # syncs gradients across nodes like ddp
                    # useful for things like increasing the number of negative samples
                    trainer = Trainer(gpus=2, num_nodes=2, distributed_backend='ddp2')

            use_amp (bool):
                .. warning:: .. deprecated:: 0.6.1
                    Use `precision` instead. Will remove 0.8.0.

            precision (int): Full precision (32), half precision (16).
                Can be used on CPU, GPU or TPUs.

                If used on TPU will use torch.bfloat16 but tensor printing
                will still show torch.float32.

                Example::

                    # default used by the Trainer
                    trainer = Trainer(precision=32)

                    # 16-bit precision
                    trainer = Trainer(precision=16)

                    # one day
                    trainer = Trainer(precision=8|4|2)

            print_nan_grads (bool): Prints gradients with nan values
                Example::

                    # default used by the Trainer
                    trainer = Trainer(print_nan_grads=False)

            weights_summary (str) or None: Prints a summary of the weights when training begins.
                Options: 'full', 'top', None.
                Example::

                    # default used by the Trainer (ie: print all weights)
                    trainer = Trainer(weights_summary='full')

                    # print only the top level modules
                    trainer = Trainer(weights_summary='top')

                    # don't print a summary
                    trainer = Trainer(weights_summary=None)

            weights_save_path (str): Where to save weights if specified.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(weights_save_path=os.getcwd())

                    # save to your custom path
                    trainer = Trainer(weights_save_path='my/path')

                    # if checkpoint callback used, then overrides the weights path
                    # **NOTE: this saves weights to some/path NOT my/path
                    checkpoint_callback = ModelCheckpoint(filepath='some/path')
                    trainer = Trainer(
                        checkpoint_callback=checkpoint_callback,
                        weights_save_path='my/path'
                    )

            amp_level (str): The optimization level to use (O1, O2, etc...).
                Check nvidia docs for level (https://nvidia.github.io/apex/amp.html#opt-levels)
                Example::

                    # default used by the Trainer
                    trainer = Trainer(amp_level='O1')

            num_sanity_val_steps (int): Sanity check runs n batches of val before starting the training routine.
                This catches any bugs in your validation without having to wait for the first validation check.
                The Trainer uses 5 steps by default. Turn it off or modify it here.
                Example::

                    # default used by the Trainer
                    trainer = Trainer(num_sanity_val_steps=5)

                    # turn it off
                    trainer = Trainer(num_sanity_val_steps=0)

            nb_sanity_val_steps (int):
                .. warning:: .. deprecated:: 0.5.0
                    Use `num_sanity_val_steps` instead. Will remove 0.8.0.

            truncated_bptt_steps (int): Truncated back prop breaks performs backprop every k steps of
                a much longer sequence If this is enabled, your batches will automatically get truncated
                and the trainer will apply Truncated Backprop to it. Make sure your batches have a sequence
                dimension. (`Williams et al. "An efficient gradient-based algorithm for on-line training of
                recurrent lightning trajectories."
                <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.7941&rep=rep1&type=pdf>`_)
                Example::

                    # default used by the Trainer (ie: disabled)
                    trainer = Trainer(truncated_bptt_steps=None)

                    # backprop every 5 steps in a batch
                    trainer = Trainer(truncated_bptt_steps=5)


                Lightning takes care to split your batch along the time-dimension.

                .. note:: If you need to modify how the batch is split,
                    override :meth:`pytorch_lightning.core.LightningModule.tbptt_split_batch`.

                .. note:: Using this feature requires updating your LightningModule's
                    :meth:`pytorch_lightning.core.LightningModule.training_step` to include a `hiddens` arg.

            resume_from_checkpoint (str): To resume training from a specific checkpoint pass in the path here.k
                Example::

                    # default used by the Trainer
                    trainer = Trainer(resume_from_checkpoint=None)

                    # resume from a specific checkpoint
                    trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
            profiler (BaseProfiler):  To profile individual steps during training and assist in
                identifying bottlenecks.
                Example::

                    from pytorch_lightning.profiler import Profiler, AdvancedProfiler

                    # default used by the Trainer
                    trainer = Trainer(profiler=None)

                    # to profile standard training events
                    trainer = Trainer(profiler=True)

                    # equivalent to profiler=True
                    profiler = Profiler()
                    trainer = Trainer(profiler=profiler)

                    # advanced profiler for function-level stats
                    profiler = AdvancedProfiler()
                    trainer = Trainer(profiler=profiler)

        .. warning:: Following arguments become deprecated and they will be removed in v0.8.0:

            - `nb_sanity_val_steps`

        """
        self.resume_cfg = resume_cfg # MANO ADDED
        # Transfer params
        # Backward compatibility
        if nb_gpu_nodes is not None:
            warnings.warn("`nb_gpu_nodes` has renamed to `num_nodes` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not num_nodes:  # in case you did not set the proper value
                num_nodes = nb_gpu_nodes
        self.num_gpu_nodes = num_nodes

        self.log_gpu_memory = log_gpu_memory

        # Backward compatibility
        if gradient_clip is not None:
            warnings.warn("`gradient_clip` has renamed to `gradient_clip_val` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not gradient_clip_val:  # in case you did not set the proper value
                gradient_clip_val = gradient_clip
        self.gradient_clip_val = gradient_clip_val

        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = True if (gpus and torch.cuda.is_available()) else False

        # tpu config
        self.on_tpu = num_tpu_cores is not None
        self.num_tpu_cores = num_tpu_cores
        assert num_tpu_cores in [1, 8, None], 'num_tpu_cores can only be 1 or 8'

        self.process_position = process_position
        self.weights_summary = weights_summary

        # Backward compatibility
        if max_nb_epochs is not None:
            warnings.warn("`max_nb_epochs` has renamed to `max_epochs` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not max_epochs:  # in case you did not set the proper value
                max_epochs = max_nb_epochs
        self.max_epochs = max_epochs

        # Backward compatibility
        if min_nb_epochs is not None:
            warnings.warn("`min_nb_epochs` has renamed to `min_epochs` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not min_epochs:  # in case you did not set the proper value
                min_epochs = min_nb_epochs
        self.min_epochs = min_epochs

        self.max_steps = max_steps
        self.min_steps = min_steps

        # Backward compatibility
        if nb_sanity_val_steps is not None:
            warnings.warn("`nb_sanity_val_steps` has renamed to `num_sanity_val_steps` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not num_sanity_val_steps:  # in case you did not set the proper value
                num_sanity_val_steps = nb_sanity_val_steps

        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.truncated_bptt_steps = truncated_bptt_steps
        self.resume_from_checkpoint = resume_from_checkpoint
        self.shown_warnings = set()

        self.fast_dev_run = fast_dev_run
        if self.fast_dev_run:
            self.num_sanity_val_steps = 1
            self.max_epochs = 1
            m = '''
            Running in fast_dev_run mode: will run a full train,
            val loop using a single batch
            '''
            log.info(m)

        # set default save path if user didn't provide one
        self.default_save_path = default_save_path
        if self.default_save_path is None:
            self.default_save_path = os.getcwd()

        # training bookeeping
        self.total_batch_idx = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_idx = 0
        self.tqdm_metrics = {}
        self.callback_metrics = {}
        self.num_val_batches = 0
        self.num_training_batches = 0
        self.num_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None
        self.is_iterable_train_dataloader = False

        # training state
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.lr_schedulers = []
        self.optimizers = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0

        # configure logger
        self.configure_logger(logger)

        # configure profiler
        if profiler is True:
            profiler = Profiler()
        self.profiler = profiler or PassThroughProfiler()

        # configure early stop callback
        # creates a default one if none passed in
        self.configure_early_stopping(early_stop_callback)

        self.reduce_lr_on_plateau_scheduler = None

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        self.weights_save_path = weights_save_path

        # accumulated grads
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.data_parallel_device_ids = parse_gpu_ids(gpus)
        self.root_gpu = determine_root_gpu_device(self.data_parallel_device_ids)

        # tpu state flags
        self.use_tpu = False
        self.tpu_local_core_rank = None
        self.tpu_global_core_rank = None

        # distributed backend choice
        self.use_ddp = False
        self.use_ddp2 = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = distributed_backend
        self.set_distributed_mode(distributed_backend, num_nodes)

        # override dist backend when using tpus
        if self.on_tpu:
            self.init_tpu()
            self.current_tpu_idx = None

        # init flags for SLURM+ddp to work
        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0
        self.configure_slurm_ddp(num_nodes)

        # nvidia setup
        self.set_nvidia_flags(self.is_slurm_managing_tasks, self.data_parallel_device_ids)

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval

        # backward compatibility
        if add_row_log_interval is not None:
            warnings.warn("`add_row_log_interval` has renamed to `row_log_interval` since v0.5.0"
                          " and this method will be removed in v0.8.0", DeprecationWarning)
            if not row_log_interval:  # in case you did not set the proper value
                row_log_interval = add_row_log_interval
        self.row_log_interval = row_log_interval

        # how much of the data to use
        self.determine_data_use_amount(train_percent_check, val_percent_check,
                                       test_percent_check, overfit_pct)

        # 16 bit mixed precision training using apex
        self.amp_level = amp_level
        self.precision = precision
        if self.precision == 16:
            use_amp = True
        self.init_amp(use_amp)

    @property
    def slurm_job_id(self):
        try:
            job_id = os.environ['SLURM_JOB_ID']
            job_id = int(job_id)
        except Exception:
            job_id = None
        return job_id

    def __parse_gpu_ids(self, gpus):
        """Parse GPUs id.

        :param list|str|int gpus: input GPU ids
        :return list(int):
        """
        # if gpus = -1 then use all available devices
        # otherwise, split the string using commas
        if gpus is not None:
            if isinstance(gpus, list):
                gpus = gpus
            elif isinstance(gpus, str):
                if gpus == '-1':
                    gpus = list(range(0, torch.cuda.device_count()))
                else:
                    gpus = [int(x.strip()) for x in gpus.split(',')]
            elif isinstance(gpus, int):
                gpus = gpus
            else:
                raise ValueError('`gpus` has to be a string, int or list of ints')

        return gpus

    def __set_root_gpu(self, gpus):
        if gpus is None:
            return None

        # set root gpu
        root_gpu = 0
        if isinstance(gpus, list):
            root_gpu = gpus[0]

        return root_gpu

    @property
    def num_gpus(self):
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0
        return len(gpus)

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp or self.use_ddp2

    @property
    def training_tqdm_dict(self):
        """Read-only for tqdm metrics.
        :return:
        """
        ref_model = self.model if not self.data_parallel else self.model.module

        return dict(**ref_model.get_tqdm_dict(), **self.tqdm_metrics)

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def fit(self, model, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        r"""
        Runs the full optimization routine.

        Args:
            model (LightningModule): Model to fit.

            train_dataloader (:class:`.torch.utils.data.DataLoader`): A Pytorch
                DataLoader with training samples. If the model has
                a predefined train_dataloader method this will be skipped.

            val_dataloader (:class:`.torch.utils.data.DataLoader`): Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloader method this will be skipped

            test_dataloader (:class:`.torch.utils.data.DataLoader`): Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined test_dataloader method this will be skipped

        Example::

            # Option 1,
            # Define the train_dataloader(), test_dataloader() and val_dataloader() fxs
            # in the lightningModule
            # RECOMMENDED FOR MOST RESEARCH AND APPLICATIONS TO MAINTAIN READABILITY
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model)

            # Option 2
            # in production cases we might want to pass different datasets to the same model
            # Recommended for PRODUCTION SYSTEMS
            train, val, test = DataLoader(...), DataLoader(...), DataLoader(...)
            trainer = Trainer()
            model = LightningModule()
            trainer.fit(model, train_dataloader=train,
                        val_dataloader=val, test_dataloader=test)

            # Option 1 & 2 can be mixed, for example the training set can be
            # defined as part of the model, and validation/test can then be
            # feed to .fit()

        """

        # Update the dataloader attributes of the model with the ones supplied here,
        # if they are not already defined in model
        _set_dataloader(model, train_dataloader, 'train_dataloader')
        _set_dataloader(model, val_dataloader, 'val_dataloader')
        _set_dataloader(model, test_dataloader, 'test_dataloader')

        # when using multi-node or DDP within a node start each module in a separate process
        if self.use_ddp2:
            task = int(os.environ['SLURM_LOCALID'])
            self.ddp_train(task, model)

        elif self.use_ddp:
            if self.is_slurm_managing_tasks:
                task = int(os.environ['SLURM_LOCALID'])
                self.ddp_train(task, model)
            else:
                mp.spawn(self.ddp_train, nprocs=self.num_gpus, args=(model,))

        # 1 gpu or dp option triggers training using DP module
        # easier to avoid NCCL issues
        elif self.use_dp:
            self.dp_train(model)

        elif self.single_gpu:
            self.single_gpu_train(model)

        elif self.use_tpu:
            log.info(f'training on {self.num_tpu_cores} TPU cores')

            #  COLAB_GPU is an env var available by default in Colab environments.
            start_method = 'fork' if os.getenv('COLAB_GPU') else 'spawn'
            xmp.spawn(self.tpu_train, args=(model,), nprocs=self.num_tpu_cores, start_method=start_method)

        # ON CPU
        else:
            # run through amp wrapper
            if self.use_amp:
                raise MisconfigurationException('amp + cpu is not supported.  Please use a GPU option')

            # CHOOSE OPTIMIZER
            # allow for lr schedulers as well
            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

            self.run_pretrain_routine(model)

        # return 1 when finished
        # used for testing or when we need to know that training succeeded
        return 1

    def init_optimizers(self, optimizers):
        # single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists
        if len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            lr_schedulers, self.reduce_lr_on_plateau_scheduler = self.configure_schedulers(lr_schedulers)
            return optimizers, lr_schedulers

        # single list or tuple
        if isinstance(optimizers, (list, tuple)):
            return optimizers, []

    def configure_schedulers(self, schedulers):
        for i, scheduler in enumerate(schedulers):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                reduce_lr_on_plateau_scheduler = schedulers.pop(i)
                return schedulers, reduce_lr_on_plateau_scheduler
        return schedulers, None

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        :param model:
        """
        ref_model = model
        if self.data_parallel:
            ref_model = model.module

        # give model convenience properties
        ref_model.trainer = self

        # set local properties on the model
        self.copy_trainer_model_properties(ref_model)

        # link up experiment object
        if self.logger is not None:
            ref_model.logger = self.logger

            # save exp to get started
            if hasattr(ref_model, "hparams"):
                self.logger.log_hyperparams(ref_model.hparams)

            self.logger.save()

        if self.use_ddp or self.use_ddp2:
            dist.barrier()

        # wait for all models to restore weights
        if self.on_tpu and XLA_AVAILABLE:
            # wait for all processes to catch up
            torch_xla.core.xla_model.rendezvous()

        # set up checkpoint callback
        self.configure_checkpoint_callback()

        # register auto-resubmit when on SLURM
        self.register_slurm_signal_handlers()

        # transfer data loaders from model
        self.get_dataloaders(ref_model)

        # print model summary
        if self.proc_rank == 0 and self.weights_summary is not None:
            if self.weights_summary in ['full', 'top']:
                ref_model.summarize(mode=self.weights_summary)
            else:
                m = "weights_summary can be None, 'full' or 'top'"
                raise MisconfigurationException(m)

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.model = model

        # restore training and model before hpc call
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            self.run_evaluation(test=True)
            return

        # check if we should run validation during training
        self.disable_validation = ((self.num_val_batches == 0 or
                                    not self.is_overriden('validation_step')) and
                                   not self.fast_dev_run)

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        ref_model.on_sanity_check_start()
        ref_model.on_train_start()
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            # init progress bars for validation sanity check
            pbar = tqdm(desc='Validation sanity check',
                        total=self.num_sanity_val_steps * len(self.get_val_dataloaders()),
                        leave=False, position=2 * self.process_position,
                        disable=not self.show_progress_bar, dynamic_ncols=True)
            self.main_progress_bar = pbar
            # dummy validation progress bar
            self.val_progress_bar = tqdm(disable=True)

            eval_results = self.evaluate(model, self.get_val_dataloaders(),
                                         self.num_sanity_val_steps, False)
            _, _, _, callback_metrics, _ = self.process_output(eval_results)

            # close progress bars
            self.main_progress_bar.close()
            self.val_progress_bar.close()

            if self.enable_early_stop:
                self.early_stop_callback.check_metrics(callback_metrics)

        # init progress bar
        pbar = tqdm(leave=True, position=2 * self.process_position,
                    disable=not self.show_progress_bar, dynamic_ncols=True,
                    file=sys.stdout)
        self.main_progress_bar = pbar

        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

        # summarize profile results
        self.profiler.describe()

    def test(self, model=None, test_dataloader=None):
        r"""

        Separates from fit to make sure you never run on your test set until you want to.

        Args:
            model (LightningModule): The model to test.

            test_dataloader (:class:`.torch.utils.data.DataLoader`): Either a single
                Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined test_loader method this will be skipped

        Example::

            # Option 1
            # run test after fitting
            trainer = Trainer()
            model = LightningModule()

            trainer.fit()
            trainer.test()

            # Option 2
            # run test from a loaded model
            model = LightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
            trainer = Trainer()
            trainer.test(model)
        """

        self.testing = True
        if model is not None:
            _set_dataloader(model, test_dataloader, 'test_dataloader')
            self.fit(model)
        else:
            self.run_evaluation(test=True)


def _set_dataloader(model, dataloader, attribute):
    r'''
    Check dataloaders passed to .fit() method if they are pytorch DataLoader
    objects and whether or not we should overright the corresponding dataloader
    in the model

    Args:
        model (LightningModule): The model to check

        dataloader: If a pytorch dataloader (or a list of pytorch dataloaders)
            is passed, it will be incorporate into the model as model.attribute.
            If attribute alreay exist it will warn the userpass. If not a
            dataloader will throw an error

        attribute (str): The attribute to save the dataloader under

    '''
    # Check if attribute comes directly from base class or
    # derived in user subclass
    if LightningModule.__qualname__ in getattr(model, attribute).__qualname__:
        # Val and test should be list of dataloaders
        dataloader = dataloader if attribute == 'train_dataloader' or \
                                   (attribute != 'train_dataloader' and isinstance(dataloader, list)) else [dataloader]

        # Check we are given valid dataloaders
        is_dataloader = isinstance(dataloader, torch.utils.data.DataLoader)
        is_dataloader_list = isinstance(dataloader, list)
        if is_dataloader_list:
            valid_loaders = all(isinstance(d, torch.utils.data.DataLoader) for d in dataloader)
        if is_dataloader or is_dataloader_list and valid_loaders:

            # Overwrite abstract methods
            dl = lambda: dataloader
            dl.__name__ = attribute
            setattr(model, attribute, dl)

        elif dataloader and dataloader != [None]:
            raise ValueError(f'`{attribute}` needs to be an instance of '
                             '`torch.utils.data.DataLoader` or a list of '
                             'DataLoaders, instead got %r`' % dataloader)

    elif dataloader:  # if default (None) is passed, do not warn the user
        pass  # MANO
        # warnings.warn(f'Model has predefined `{attribute}`,'
        #               f' will skip `{attribute}={dataloader}` passed to fit method.')
