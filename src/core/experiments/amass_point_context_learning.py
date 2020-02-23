from util.torch.nn import set_determinsitic_run
from dataset.datasets import FullPartDatasetMenu
from util.strings import banner, set_logging_to_stdout
from util.torch.data import none_or_int, none_or_str
from test_tube import HyperOptArgumentParser
from architecture.models import PointContextNet
from lightning.completion_net import init_trainer
from dataset.transforms import *

set_logging_to_stdout()
set_determinsitic_run()  # Set a universal random seed


# ----------------------------------------------------------------------------------------------------------------------
#                                               Main Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parser():
    p = HyperOptArgumentParser(strategy='random_search')
    # Check-pointing
    # TODO - Don't forget to change me!
    p.add_argument('--exp_name', type=str, default='amass_point_context_learning', help='The experiment name. Leave empty for default')
    p.add_argument('--resume_version', type=none_or_int, default=None,
                   help='Try train resume of exp_name/version_{resume_version} checkpoint. Use None for no resume')
    p.add_argument('--save_completions', type=int, choices=[0, 1, 2, 3], default=2,
                   help='Use 0 for no save. Use 1 for vertex only save in obj file. Use 2 for a full mesh save (v&f). '
                        'Use 3 for gt,tp,gt_part,tp_part save as well.')

    # Architecture
    p.add_argument('--dense_encoder', type=bool, default=False, help='If true uses dense encoder architecture')
    p.add_argument('--use_default_init', type=bool,default=False, help='If true, using default kaiming init. Else - '
                                                                      'using our .init_weights()')

    # Dataset Config:
    # NOTE: A well known ML rule: double the learning rate if you double the batch size.
    p.add_argument('--batch_size', type=int, default=10, help='SGD batch size')
    p.add_argument('--counts', nargs=3, type=none_or_int, default=(10000, 1000, 1000), # TODO - If amass - edit this
                   help='[Train,Validation,Test] number of samples. Use None for all in partition')
    p.add_argument('--in_channels', choices=[3, 6, 12], default=6,
                   help='Number of input channels')

    # Train Config:
    p.add_argument('--force_train_epoches', type=int, default=1,
                   help="Force train for this amount. Usually we'd early stop using the callback. Use 1 to disable")
    p.add_argument('--lr', type=float, default=0.001, help='The learning step to use')

    # Optimizer
    p.add_argument("--weight_decay", type=float, default=0, help="Adam's weight decay - usually use 1e-4")
    p.add_argument("--plateau_patience", type=none_or_int, default=None,
                   help="Number of epoches to wait on learning plateau before reducing step size. Use None to shut off")
    p.add_argument("--early_stop_patience", type=int, default=100,
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
    p.add_argument('--loss_class', type=str, choices=['BasicLoss', 'SkepticLoss', 'TBasedLoss', 'PointContextLoss'], default='PointContextLoss',
                   help='The loss class')
    # TODO - is this the right way to go?

    # Computation
    p.add_argument('--gpus', type=none_or_int, default=-1, help='Use -1 to use all available. Use None to run on CPU')
    p.add_argument('--distributed_backend', type=str, default='dp',
                   help='supports three options dp, ddp, ddp2')  # TODO - ddp2,ddp Untested
    p.add_argument('--use_16b', type=bool, default=False, help='If true uses 16 bit precision')  # TODO - Untested

    # Visualization
    p.add_argument('--use_auto_tensorboard', type=bool, default=1,
                   help='Mode: 0 - Does nothing. 1 - Opens up only server. 2 - Opens up only chrome. 3- Opens up both '
                        'chrome and server')
    p.add_argument('--use_logger', type=bool, default=True,  # TODO - Not in use
                   help='Whether to log information or not')
    p.add_argument('--plotter_class', type=none_or_str, choices=[None, 'CompletionPlotter'],
                   default='CompletionPlotter',
                   help='The plotter class or None for no plot')  # TODO - is this the right way to go?

    return [p]


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Mains
# ----------------------------------------------------------------------------------------------------------------------
def train_main():
    banner('Network Init')
    nn = PointContextNet(parser())  # TODO - Change me Oshri
    nn.identify_system()

    hp = nn.hyper_params()

    tr_ds = FullPartDatasetMenu.get('AmassTrainPyProj')
    tr_ldr = tr_ds.loaders(s_dynamic=True, s_nums=hp.counts[0], s_transform=[Center()], batch_size=hp.batch_size, device=hp.dev,
                           n_channels=hp.in_channels, method='rand_f2p')
    val_ds = FullPartDatasetMenu.get('AmassValdPyProj')
    val_ldr = val_ds.loaders(s_nums=hp.counts[1], s_transform=[Center()], batch_size=hp.batch_size, device=hp.dev,
                             n_channels=hp.in_channels, method='rand_f2p') # Should have been f2p, but it is too big
    ts_ds = FullPartDatasetMenu.get('AmassTestPyProj')
    ts_ldr = ts_ds.loaders(s_nums=hp.counts[2], s_transform=[Center()], batch_size=hp.batch_size, device=hp.dev,
                           n_channels=hp.in_channels, method='f2p')

    nn.init_data(loaders=[tr_ldr, val_ldr, ts_ldr])

    trainer = init_trainer(nn, fast_dev_run=False)
    banner('Training Phase')
    trainer.fit(nn)
    banner('Testing Phase')
    trainer.test(nn)
    nn.finalize()


if __name__ == '__main__':
    train_main()
