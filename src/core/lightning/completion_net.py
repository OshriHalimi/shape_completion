import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # , CosineAnnealingLR
import architecture.loss
from util.mesh.ops import batch_vnrmls
from util.torch.nn import PytorchNet
from util.func import all_variables_by_module_name
from copy import deepcopy


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class CompletionLightningModel(PytorchNet):
    def __init__(self, hp=()):
        super().__init__()
        self.hparams = self.add_model_specific_args(hp).parse_args()
        self.hp = self.hparams  # Aliasing
        self._append_config_args()  # Must be done here, seeing we need hp.dev

        # Bookeeping:
        self.assets = None  # Set by Trainer
        self.loss, self.opt = None, None

        self._build_model()
        self.type(dst_type=getattr(torch, self.hparams.UNIVERSAL_PRECISION))  # Transfer to precision
        self._init_model()

    @staticmethod  # You need to override this method
    def add_model_specific_args(parent_parser):
        return parent_parser

    def forward(self, input_dict):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def complete(self, input_dict):

        output_dict = self.forward(input_dict)
        # TODO - Implement Generic Loss and fix this function
        if self.hparams.compute_output_normals:
            vnb, vnb_is_valid = batch_vnrmls(output_dict['completion_xyz'], self.assets.data.torch_faces(),
                                             return_f_areas=False)
            output_dict['completion_vnb'] = vnb
            output_dict['completion_vnb_is_valid'] = vnb_is_valid

        return output_dict

    def configure_optimizers(self):
        loss_cls = getattr(architecture.loss, self.hp.loss_class)
        self.loss = loss_cls(hp=self.hp, f=self.assets.data.faces())
        self.opt = torch.optim.Adam(self.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)

        if self.hp.plateau_patience is not None:
            # sched = CosineAnnealingLR(optimizer, T_max=10)
            sched = ReduceLROnPlateau(self.opt, mode='min', patience=self.hp.plateau_patience, verbose=True,
                                      cooldown=self.hp.DEF_LR_SCHED_COOLDOWN, eps=self.hp.DEF_MINIMAL_LR)
            # Options: factor=0.1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
            return [self.opt], [sched]
        else:
            return [self.opt]

    def training_step(self, b, batch_idx):
        completion = self.complete(b)
        loss_dict = self.loss.compute(b, completion)
        loss_dict = {f'{k}_train': v for k, v in loss_dict.items()}  # make different logs for train, test, validation
        train_loss = loss_dict['total_loss_train']

        if self.assets.plt and batch_idx == 0:  # On first batch
            self.assets.plt.cache(self.assets.plt.prepare_plotter_dict(b, completion))  # New tensors, without grad

        return {
            'loss': train_loss,  # Must use 'loss' instead of 'train_loss' due to old_lightning framework
            'log': loss_dict
        }

    def validation_step(self, b, batch_idx, set_id=0):
        pred = self.complete(b)

        if self.assets.plt and batch_idx == 0 and set_id == 0:  # On first batch, of first dataset. TODO - Generalize
            new_data = (self.assets.plt.uncache(), self.assets.plt.prepare_plotter_dict(b, pred))
            self.assets.plt.push(new_data=new_data, new_epoch=self.current_epoch)

        return self.loss.compute(b, pred)

    def validation_end(self, outputs):

        if self.assets.data.num_vald_loaders() == 1:
            outputs = [outputs] # Incase singleton case
        log_dict, progbar_dict = {}, {}
        avg_val_loss = 0
        for i in range(len(outputs)):  # Number of validation datasets
            ds_name = self.assets.data.id2vald_ds(i)
            for k in outputs[i][0].keys():
                log_dict[f'{k}_val_{ds_name}'] = torch.stack([x[k] for x in outputs[i]]).mean()
            ds_val_loss = log_dict[f'total_loss_val_{ds_name}']
            progbar_dict[f'val_loss_{ds_name}'] = ds_val_loss
            if i == 0:  # Always use the first dataset as the validation loss
                avg_val_loss = ds_val_loss
                progbar_dict['val_loss'] = avg_val_loss

        lr = self.learning_rate(self.opt)  # Also log learning rate
        progbar_dict['lr'], log_dict['lr'] = lr, lr

        # This must be kept as "val_loss" and not "avg_val_loss" due to old_lightning bug
        return {"val_loss": avg_val_loss,  # TODO - Remove double entry for val_koss
                "progress_bar": progbar_dict,
                "log": log_dict}

    def test_step(self, b, _, set_id=0):

        pred = self.complete(b)
        if self.assets.saver:  # TODO - Generalize this
            self.assets.saver.save_completions_by_batch(pred, b, set_id)
        return self.loss.compute(b, pred)

    def test_end(self, outputs):
        if self.assets.data.num_test_loaders() == 1:
            outputs = [outputs] # Incase singleton case
        log_dict, progbar_dict = {}, {}
        avg_test_loss = 0
        for i in range(len(outputs)):  # Number of test datasets
            ds_name = self.assets.data.id2test_ds(i)
            for k in outputs[i][0].keys():
                log_dict[f'{k}_test_{ds_name}'] = torch.stack([x[k] for x in outputs[i]]).mean()
            ds_test_loss = log_dict[f'total_loss_test_{ds_name}']
            progbar_dict[f'test_loss_{ds_name}'] = ds_test_loss
            if i == 0:  # Always use the first dataset as the test loss
                avg_test_loss = ds_test_loss
                progbar_dict['test_loss'] = avg_test_loss

        return {"test_loss": avg_test_loss,
                "progress_bar": progbar_dict,
                "log": log_dict}

    def hyper_params(self):
        return deepcopy(self.hp)

    def _append_config_args(self):

        for k, v in all_variables_by_module_name('cfg').items():  # Only import non-class/module types
            setattr(self.hp, k, v)

        # Architecture name:
        setattr(self.hp, 'arch', self.family_name())

        if hasattr(self.hp, 'gpus'):  # This is here to allow init of lightning with only model params (no argin)

            # Device - TODO - Does this support multiple GPU ?
            dev = torch.device('cpu') if self.hp.gpus is None else torch.device('cuda', torch.cuda.current_device())
            setattr(self.hp, 'dev', dev)

            # Experiment:
            if self.hp.exp_name is None or not self.hp.exp_name:
                self.hp.exp_name = 'default_exp'

            # Epochs:
            if self.hp.max_epochs is None:
                self.hp.max_epochs = 10000000

            # Correctness of config parameters:
            assert self.hp.VIS_N_MESH_SETS <= self.hp.batch_size, \
                f"Plotter needs requires batch size >= N_MESH_SETS={self.hp.VIS_N_MESH_SETS}"
