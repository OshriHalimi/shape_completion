from util.torch_nn import PytorchNet, set_determinsitic_run
from dataset.datasets import FullPartDatasetMenu
from util.string_op import banner, set_logging_to_stdout
from util.func import tutorial
from util.torch_data import none_or_int, none_or_str
from test_tube import HyperOptArgumentParser
from architecture.models import F2PEncoderDecoder
from architecture.lightning import lightning_trainer, test_lightning
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
    p.add_argument('--exp_name', type=str, default='test_code', help='The experiment name. Leave empty for default')
    p.add_argument('--resume_version', type=none_or_int, default=0,
                   help='Try train resume of exp_name/version_{resume_version} checkpoint. Use None for no resume')
    p.add_argument('--save_completions', type=int, choices=[0, 1, 2,3], default=2,
                   help='Use 0 for no save. Use 1 for vertex only save in obj file. Use 2 for a full mesh save (v&f). '
                        'Use 3 for gt,tp,gt_part,tp_part save as well.')

    # Architecture
    p.add_argument('--dense_encoder', type=bool, default=True, help='If true uses dense encoder architecture')

    # Dataset Config:
    # NOTE: A well known ML rule: double the learning rate if you double the batch size.
    p.add_argument('--batch_size', type=int, default=3, help='SGD batch size')
    p.add_argument('--counts', nargs=3, type=none_or_int, default=(10, 10, 10),
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

    # L2 Losses: Use 0 to ignore, >0 to compute
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
                   help='The loss class')
    # TODO - is this the right way to go?

    # Computation
    p.add_argument('--gpus', type=none_or_int, default=-1, help='Use -1 to use all available. Use None to run on CPU')
    p.add_argument('--distributed_backend', type=str, default='dp',
                   help='supports three options dp, ddp, ddp2')  # TODO - ddp2,ddp Untested
    p.add_argument('--use_16b', type=bool, default=False, help='If true uses 16 bit precision')  # TODO - Untested

    # Visualization
    p.add_argument('--use_auto_tensorboard', type=bool, default=False,
                   help='Whether to automatically open up the tensorboard server and chrome process')
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
    nn = F2PEncoderDecoder(parser())
    nn.identify_system()

    hp = nn.hyper_params()
    # Init loaders and faces:
    ds = FullPartDatasetMenu.get('FaustPyProj')
    ldrs = ds.split_loaders(split=[0.8, 0.1, 0.1], s_nums=hp.counts, s_shuffle=[True] * 3, s_transform=[Center()] * 3,
                            batch_size=hp.batch_size, device=hp.dev, n_channels=hp.in_channels, method='f2p',
                            s_dynamic=[False] * 3)
    nn.init_data(loaders=ldrs)

    trainer = lightning_trainer(nn, fast_dev_run=False)
    banner('Training Phase')
    trainer.fit(nn)
    banner('Testing Phase')
    trainer.test(nn)
    nn.finalize()


def test_main():
    # TODO - Fix this
    nn = F2PEncoderDecoder(parser())
    hp = nn.hyper_params()
    ds = FullPartDatasetMenu.get('DFaustPyProj')
    test_ldr = ds.split_loaders(s_nums=hp.counts[2], s_transform=[Center()], batch_size=hp.batch_size,
                                device=hp.dev, n_channels=hp.in_channels, method='f2p')
    nn.init_data(loaders=[None, None, test_ldr])
    test_lightning(nn)


# ----------------------------------------------------------------------------------------------------------------------
#                                                 Tutorials
# ----------------------------------------------------------------------------------------------------------------------
@tutorial
def dataset_tutorial():
    # Use the menu to see which datasets are implemented
    print(FullPartDatasetMenu.which())
    ds = FullPartDatasetMenu.get('FaustPyProj')  # This will fail if you don't have the data on disk
    ds.validate_dataset()  # Make sure all files are available - Only run this once, to make sure.

    banner('The HIT')
    ds.report_index_tree()  # Take a look at how the dataset is indexed - using the hit [HierarchicalIndexTree]

    banner('Collateral Info')
    print(f'Dataset Name = {ds.name()}')
    print(f'Number of indexed files = {ds.num_indexed()}')
    print(f'Number of full shapes = {ds.num_full_shapes()}')
    print(f'Number of projections = {ds.num_projections()}')
    print(f'Required disk space in bytes = {ds.disk_space()}')
    # You can also request a summary printout with:
    ds.data_summary(with_tree=False)  # Don't print out the tree again

    # For models with a single set of faces (SMPL or SMLR for example) you can request the face set/number of vertices
    # directly:
    banner('Face Array')
    print(ds.faces())
    print(ds.num_faces())
    print(ds.num_verts())
    # You can also ask for the null-shape the dataset - with hi : [0,0...,0]
    print(ds.null_shape(n_channels=6))
    ds.plot_null_shape(strategy='spheres', with_vnormals=True)

    # Let's look at the various sampling methods available to us:
    print(ds.defined_methods())  # ('full', 'part', 'f2p', 'rand_f2p', 'p2p', 'rand_p2p')
    # We can ask for a sample of the data with this sampling method:
    banner('Data Sample')
    samp = ds.sample(num_samples=2, transforms=[Center(keys=['gt'])], n_channels=6, method='full')
    print(samp)  # Dict with gt_hi & gt
    print(ds.num_datapoints_by_method('full'))  # 100

    samp = ds.sample(num_samples=2, transforms=[Center(keys=['gt'])], n_channels=6, method='part')
    print(samp)  # Dict with gt_hi & gt & gt_mask & gt_mask
    print(ds.num_datapoints_by_method('part'))  # 1000

    samp = ds.sample(num_samples=2, transforms=[Center(keys=['gt'])], n_channels=6, method='f2p')
    print(samp)  # Dict with gt_hi & gt & gt_mask & gt_mask & tp
    print(ds.num_datapoints_by_method('f2p'))  # 10000 tuples of (gt,tp) where the subjects are the same

    # # You can also ask for a simple loader, given by the ids you'd like to see.
    # # Pass ids = None to index the entire dataset, form point_cloud = 0 to point_cloud = num_datapoints_by_method -1
    banner('Loaders')
    single_ldr = ds.split_loaders(s_nums=1000, s_shuffle=True, s_transform=[Center()], n_channels=6, method='f2p',
                                  batch_size=3, device='cpu-single')
    for d in single_ldr:
        print(d)
        break

    print(single_ldr.num_verts())
    # There are also operations defined on the loaders themselves. See utils.torch_data for details

    # To receive train/validation splits or train/validation/test splits use:
    my_loaders = ds.split_loaders(split=[0.8, 0.1, 0.1], s_nums=[2000, 1000, 1000],
                                  s_shuffle=[True] * 3, s_transform=[Center()] * 3, global_shuffle=True, method='p2p',
                                  s_dynamic=[True, False, False])

    # Please read the documentation of split_loaders for the exact details. In essence:
    # You'll receive len(split) dataloaders, where each part i is split[i]*num_point_clouds size. From this split,
    # s_nums[i] will be taken for the dataloader, and transformed by s_transform[i].
    # s_shuffle and global_shuffle controls the shuffling of the different partitions - see doc inside function


@tutorial
def pytorch_net_tutorial():
    # What is it? PyTorchNet is a derived class of LightningModule, allowing for extended operations on it
    # Let's see some of them:
    nn = F2PEncoderDecoder()  # Remember that F2PEncoderDecoder is a subclass of PytorchNet
    nn.identify_system()  # Outputs the specs of the current system - Useful for identifying existing GPUs

    banner('General Net Info')
    target_input_size = ((6890, 3), (6890, 3))
    nn.summary(x_shape=target_input_size)
    nn.summary(batch_size=3, x_shape=target_input_size)
    print(f'On GPU = {nn.ongpu()}')  # Whether the system is on the GPU or not. Will print False
    nn.print_memory_usage(device=0)  # Print GPU 0's memory consumption
    print(f'Output size = {nn.output_size(x_shape=target_input_size)}')
    nn.print_weights()
    nn.visualize(x_shape=target_input_size, frmt='pdf')  # Prints PDF to current directory

    # Let's say we have some sort of nn.Module network:
    banner('Some network from the net usecase')
    import torchvision
    nn = torchvision.models.resnet50(pretrained=False)
    # We can extend it's functionality at runtime with a monkeypatch:
    py_nn = PytorchNet.monkeypatch(nn)
    py_nn.summary(x_shape=(3, 28, 28), batch_size=64)


@tutorial
def shortcuts_tutorial():
    print("""
    Existing shortcuts are: 
    r = reconstruction
    b = batch
    v = vertex
    d = dict / dir 
    s = string or split 
    vn = vertex normals
    n_v = num vertices
    f = face / file
    n_f = num faces 
    fn = face normals 
    gt = ground truth
    tp = template
    i = index 
    p = path 
    fp = file path 
    dp = directory path 
    hp = hyper parameters 
    ds = dataset 
    You can also concatenate - gtrb = Ground Truth Reconstruction Batched  
    """)


if __name__ == '__main__':
    train_main()
