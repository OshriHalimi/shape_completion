from lightning.trainer import LightningTrainer
from util.torch.nn import set_determinsitic_run
from util.torch.data import none_or_int, none_or_str
from util.strings import banner, set_logging_to_stdout
from dataset.datasets import FullPartDatasetMenu
from dataset.transforms import *
from architecture.models import *

set_logging_to_stdout()
set_determinsitic_run()  # Set a universal random seed


# ----------------------------------------------------------------------------------------------------------------------
#                                               Main Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parser():
    p = HyperOptArgumentParser(strategy='random_search')

    # Check-pointing
    p.add_argument('--exp_name', type=str, default='mixamo_cross_dataset_realistic_basic',  # TODO - Don't forget to change me!
                   help='The experiment name. Leave empty for default')
    p.add_argument('--version', type=none_or_int, default=0,
                   help='Weights will be saved at weight_dir=exp_name/version_{version}. '
                        'Use None to automatically choose an unused version')
    p.add_argument('--resume_cfg', nargs=2, type=bool, default=(False, True),
                   help='Only works if version != None and and weight_dir exists. '
                        '1st Bool: Whether to attempt restore of early stopping callback. '
                        '2nd Bool: Whether to attempt restore learning rate scheduler')
    p.add_argument('--save_completions', type=int, choices=[0, 1, 2, 3], default=2,
                   help='Use 0 for no save. Use 1 for vertex only save in obj file. Use 2 for a full mesh save (v&f). '
                        'Use 3 for gt,tp,gt_part,tp_part save as well.')

    # Dataset Config:
    # NOTE: A well known ML rule: double the learning rate if you double the batch size.
    p.add_argument('--batch_size', type=int, default=10, help='SGD batch size')
    p.add_argument('--in_channels', choices=[3, 6, 12], default=6,
                   help='Number of input channels')

    # Train Config:
    p.add_argument('--force_train_epoches', type=int, default=1,
                   help="Force train for this amount. Usually we'd early stop using the callback. Use 1 to disable")
    p.add_argument('--max_epochs', type=int, default=None,  # Must be over 1
                   help='Maximum epochs to train for. Use None for close to infinite epochs')
    p.add_argument('--lr', type=float, default=0.001, help='The learning step to use')

    # Optimizer
    p.add_argument("--weight_decay", type=float, default=0, help="Adam's weight decay - usually use 1e-4")
    p.add_argument("--plateau_patience", type=none_or_int, default=None,
                   help="Number of epoches to wait on learning plateau before reducing step size. Use None to shut off")
    p.add_argument("--early_stop_patience", type=int, default=80,  # TODO - Remember to setup resume_cfg correctly
                   help="Number of epoches to wait on learning plateau before stopping train")
    # Without early stop callback, we'll train for cfg.MAX_EPOCHS

    # L2 Losses: Use 0 to ignore, >0 to lightning
    p.add_argument('--lambdas', nargs=7, type=float, default=(1, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,EuclidDistMat,EuclidNormalDistMap,FaceAreas,Volume]'
                        'loss multiplication modifiers')
    p.add_argument('--mask_penalties', nargs=7, type=float, default=(0, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,EuclidDistMat,EuclidNormalDistMap,FaceAreas,Volume]'
                        'increased weight on mask vertices. Use val <= 1 to disable')
    p.add_argument('--dist_v_penalties', nargs=7, type=float, default=(0, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,EuclidDistMat,EuclidNormalDistMap, FaceAreas, Volume]'
                        'increased weight on distant vertices. Use val <= 1 to disable')
    p.add_argument('--loss_class', type=str, choices=['BasicLoss', 'SkepticLoss'], default='BasicLoss',
                   help='The loss class')  # TODO - generalize this

    # Computation
    p.add_argument('--gpus', type=none_or_int, default=-1, help='Use -1 to use all available. Use None to run on CPU')
    p.add_argument('--distributed_backend', type=str, default='dp', help='supports three options dp, ddp, ddp2')
    # TODO - ddp2,ddp Untested. Multiple GPUS - not tested

    # Visualization
    p.add_argument('--use_auto_tensorboard', type=bool, default=3,
                   help='Mode: 0 - Does nothing. 1 - Opens up only server. 2 - Opens up only chrome. 3- Opens up both '
                        'chrome and server')
    p.add_argument('--plotter_class', type=none_or_str, choices=[None, 'CompletionPlotter'],
                   default='CompletionPlotter',
                   help='The plotter class or None for no plot')  # TODO - generalize this

    # Completion Report
    p.add_argument('--email_report', type=bool, default=True,
                   help='Email basic tensorboard dir if True')

    return [p]


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Mains
# ----------------------------------------------------------------------------------------------------------------------
def train_main():
    banner('Network Init')
    nn = F2PEncoderDecoderRealistic(parser())
    nn.identify_system()

    # Bring in data:
    ldrs = mixamo_loader_set(nn.hp)

    # Supply the network with the loaders:
    trainer = LightningTrainer(nn, ldrs)
    trainer.train()
    trainer.test()
    trainer.finalize()


def mixamo_loader_set(hp):
    # TODO - remember to change me path to Mixamo.
    ds_mixamo = FullPartDatasetMenu.get('MixamoPyProj', data_dir_override="Z:\ShapeCompletion\Mixamo")
    ldrs = ds_mixamo.loaders(split=[0.8, 0.1, 0.1], s_nums=[10000, 1000, 1000], s_shuffle=[True] * 3,
                             s_transform=[Center()] * 3, batch_size=hp.batch_size, device=hp.dev,
                             n_channels=hp.in_channels, method='rand_f2p', s_dynamic=[True, False, False])
    ldrs[1], ldrs[2] = [ldrs[1]], [ldrs[2]]

    ds = FullPartDatasetMenu.get('FaustPyProj')
    # MIXAMO is composed from Faust subjects - Do not use them in the test/validation due to contamination
    tv_ldrs = ds.loaders(split=[0.8,0.1,0.1], s_nums=[1000]*3, s_transform=[Center()] * 3,
                         batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels,
                         method='f2p', s_shuffle=[True] * 3, s_dynamic=[False]*3)
    ldrs[1].append(tv_ldrs[1]), ldrs[2].append(tv_ldrs[2])

    ds = FullPartDatasetMenu.get('DFaustPyProj')
    tv_ldrs = ds.loaders(split=[0.2, 0.8], s_nums=[1000, 1000], s_transform=[Center()] * 2,
                         batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels,
                         method='rand_f2p', s_shuffle=[True] * 2, s_dynamic=[False, False])
    ldrs[1].append(tv_ldrs[0]), ldrs[2].append(tv_ldrs[1])

    # ds = FullPartDatasetMenu.get('DFaustPyProjSeq',
    #                              data_dir_override=hp.PRIMARY_DATA_DIR / 'synthetic' / 'DFaustPyProj')
    # tv_ldrs = ds.loaders(split=[0.2, 0.8], s_nums=[1000, 1000], s_transform=[Center()] * 2,
    #                      batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels,
    #                      method='rand_f2p_seq', s_shuffle=[True] * 2, s_dynamic=[False, False])
    # ldrs[1].append(tv_ldrs[0]), ldrs[2].append(tv_ldrs[1])

    # ds = FullPartDatasetMenu.get('AmassValdPyProj')  # AmassTestPyProj sucks
    # tv_ldrs = ds.loaders(split=[0.2, 0.8], s_nums=[1000, 1000], s_transform=[Center()] * 2,
    #                      batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels,
    #                      method='rand_f2p', s_shuffle=[True] * 2, s_dynamic=[False, False])
    # ldrs[1].append(tv_ldrs[0]), ldrs[2].append(tv_ldrs[1])

    ds = FullPartDatasetMenu.get('AmassTrainPyProj')
    tv_ldrs = ds.loaders(split=[0.2, 0.8], s_nums=[1000, 1000], s_transform=[Center()] * 2,
                         batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels,
                         method='rand_f2p', s_shuffle=[True] * 2, s_dynamic=[False, False])
    ldrs[1].append(tv_ldrs[0]), ldrs[2].append(tv_ldrs[1])

    return ldrs


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    train_main()
