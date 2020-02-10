import os
from abc import ABC

from lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.logging import TestTubeLogger


class TrainerCallbackConfigMixin(ABC):

    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.default_save_path = None
        self.save_checkpoint = None
        self.slurm_job_id = None

    def configure_checkpoint_callback(self):
        """
        Weight path set in this priority:
        Checkpoint_callback's path (if passed in).
        User provided weights_saved_path
        Otherwise use os.getcwd()
        """
        if self.checkpoint_callback is True:
            # init a default one
            if self.logger is not None:
                ckpt_path = os.path.join(
                    self.default_save_path,
                    self.logger.name,
                    f'version_{self.logger.version}',
                    "checkpoints"
                )
            else:
                ckpt_path = os.path.join(self.default_save_path, "checkpoints")

            self.checkpoint_callback = ModelCheckpoint(
                filepath=ckpt_path
            )
        elif self.checkpoint_callback is False:
            self.checkpoint_callback = None

        if self.checkpoint_callback:
            # set the path for the callbacks
            self.checkpoint_callback.save_function = self.save_checkpoint

            # if checkpoint callback used, then override the weights path
            self.weights_save_path = self.checkpoint_callback.filepath

        # if weights_save_path is still none here, set to current working dir
        if self.weights_save_path is None:
            self.weights_save_path = self.default_save_path

    def configure_early_stopping(self, early_stop_callback, logger):
        if early_stop_callback is True:
            self.early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=True,
                mode='min'
            )
            self.enable_early_stop = True
        elif not early_stop_callback:
            self.early_stop_callback = None
            self.enable_early_stop = False
        else:
            self.early_stop_callback = early_stop_callback
            self.enable_early_stop = True

        # configure logger
        if logger is True:
            # default logger
            self.logger = TestTubeLogger(
                save_dir=self.default_save_path,
                version=self.slurm_job_id,
                name='lightning_logs'
            )
            self.logger.rank = 0
        elif logger is False:
            self.logger = None
        else:
            self.logger = logger
            self.logger.rank = 0
