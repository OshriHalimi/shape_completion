from util.torch_nn import PytorchNet, set_determinsitic_run
from dataset.datasets import PointDatasetMenu
from util.string_op import banner, set_logging_to_stdout
from util.torch_data import none_or_int, none_or_str
from test_tube import HyperOptArgumentParser
from architecture.models import F2PEncoderDecoderSkeptic
from architecture.lightning import lightning_trainer
from dataset.transforms import *
from dataset.abstract import InCfg

set_logging_to_stdout()
set_determinsitic_run()  # Set a universal random seed


# ----------------------------------------------------------------------------------------------------------------------
#                                               Main Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parser():
    p = HyperOptArgumentParser(strategy='random_search')
    # Check-pointing
    # TODO - Don't forget to change me!
    p.add_argument('--exp_name', type=str, default='skeptic_architecture_exp', help='The experiment name. Leave empty for default')
    p.add_argument('--resume_version', type=none_or_int, default=None, #TODO: resume is not working! It seems to write to the requested file but the training starts from Epoch=0 and high loss (previous weights are not loaded)
                   help='Try train resume of exp_name/version_{resume_version} checkpoint. Use None for no resume')
    p.add_argument('--save_completions', type=int, choices=[0, 1, 2], default=2,
                   help='Use 0 for no save. Use 1 for vertex only save in obj file. Use 2 for a full mesh save (v&f)')

    # Architecture
    p.add_argument('--dense_encoder', type=bool, default=False, help='If true uses dense encoder architecture')

    # Dataset Config:
    # NOTE: A well known ML rule: double the learning rate if you double the batch size.
    p.add_argument('--batch_size', type=int, default=10, help='SGD batch size')
    p.add_argument('--counts', nargs=3, type=none_or_int, default=(None, None, None),
                   help='[Train,Validation,Test] number of samples. Use None for all in partition')
    p.add_argument('--in_channels', choices=[3, 6, 12], default=6,
                   help='Number of input channels')

    # Train Config:
    p.add_argument('--force_train_epoches', type=int, default=1,
                   help="Force train for this amount. Usually we'd early stop using the callback. Use 1 to disable")

    # Optimizer
    p.add_argument('--lr', type=float, default=0.001, help='The learning step to use')
    p.add_argument("--weight_decay", type=float, default=0, help="Adam's weight decay - usually use 1e-4")
    p.add_argument("--plateau_patience", type=none_or_int, default=None,
                   help="Number of epoches to wait on learning plateau before reducing step size. Use None to shut off")
    p.add_argument("--early_stop_patience", type=int, default=100,
                   help="Number of epoches to wait on learning plateau before stopping train")
    # Without early stop callback, we'll train for cfg.MAX_EPOCHS


    # Loss
    p.add_argument('--loss_class', type=str, choices=['BasicLoss', 'SkepticLoss'], default='SkepticLoss',
                   help='The loss class')
    # Shape diff Losses: Use 0 to ignore, >0 to compute
    p.add_argument('--lambdas', nargs=4, type=float, default=(1, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,Euclid_distortion, Euclid_distortion normals,FaceAreas, Volume] loss multiplication modifiers')
    p.add_argument('--mask_penalties', nargs=3, type=float, default=(0, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,Euclid_distortion, Euclid_distortion normals, FaceAreas, Volume] increased weight on mask vertices. Use val <= 1 to disable')
    p.add_argument('--dist_v_penalties', nargs=3, type=float, default=(0, 0, 0, 0, 0, 0, 0),
                   help='[XYZ,Normal,Moments,Euclid_distortion, Euclid_distortion normals, FaceAreas, Volume] increased weight on distant vertices. Use val <= 1 to disable')

    # Computation
    p.add_argument('--gpus', type=none_or_int, default=-1, help='Use -1 to use all available. Use None to run on CPU')
    p.add_argument('--distributed_backend', type=str, default='dp',
                   help='supports three options dp, ddp, ddp2')  # TODO - ddp2,ddp Untested
    p.add_argument('--use_16b', type=bool, default=False, help='If true uses 16 bit precision')  # TODO - Untested

    # Visualization
    p.add_argument('--use_tensorboard', type=bool, default=True,  # TODO - Not in use
                   help='Whether to log information to tensorboard or not')
    p.add_argument('--parallel_plotter', type=none_or_str, choices=[None, 'CompletionPlotter'], default='CompletionPlotter',
                   help='The plotter class or None for no plot')

    return [p]


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Mains
# ----------------------------------------------------------------------------------------------------------------------
def train_main():
    banner('Network Init')
    nn = F2PEncoderDecoderSkeptic(parser())
    nn.identify_system()

    hp = nn.hyper_params()
    # Init loaders and faces:
    ds = PointDatasetMenu.get('FaustPyProj', in_cfg=InCfg.FULL2PART, in_channels=hp.in_channels)
    ldrs = ds.split_loaders(split=[0.8, 0.1, 0.1], s_nums=hp.counts,
                            s_shuffle=[True] * 3, s_transform=[Center()] * 3, batch_size=hp.batch_size, device=hp.dev)
    nn.init_data(loaders=ldrs)

    trainer = lightning_trainer(nn, fast_dev_run=False)
    banner('Training Phase')
    trainer.fit(nn)
    banner('Testing Phase')
    trainer.test(nn)

if __name__ == '__main__':
    train_main()
