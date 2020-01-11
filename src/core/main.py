from dataset.datasets import PointDatasetMenu
from util.gen import none_or_int, banner, tutorial
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from architecture.models import F2PEncoderDecoder
from architecture.pytorch_extensions import PytorchNet, set_determinsitic_run
from dataset.transforms import *

# ----------------------------------------------------------------------------------------------------------------------
#                                               Main Arguments
# ----------------------------------------------------------------------------------------------------------------------
def parse_args(model_cls):
    p = HyperOptArgumentParser(strategy='random_search')
    # Exp Config:
    p.add_argument('--exp_name', type=str, default='General',
                   help='The experiment name. Leave blank to use the generic exp name')

    # NN Config
    p.add_argument('--resume_by', choices=['', 'val_acc', 'time'], default='time',
                   help='Resume Configuration - Either by (1) Latest time OR (2) Best Vald Acc OR (3) No Resume')
    p.add_argument('--use_visdom', type=bool, default=True)

    # Dataset Config:
    p.add_argument('--data_samples', nargs=3, type=none_or_int, default=[None, 1000, 1000],
                   help='[Train,Validation,Test] number of samples. Use None for all in partition')
    p.add_argument('--in_channels', choices=[3, 6, 12], default=6,
                   help='Number of input channels')

    # Train Config:
    p.add_argument('--batch_size', type=int, default=16, help='SGD batch size')
    p.add_argument('--n_epoch', type=int, default=1000, help='The number of epochs to train for')

    # Losses: Use 0 to ignore, >0 to compute
    p.add_argument('--lambdas', nargs=4, type=float, default=[1, 0.1, 0, 0],
                   help='[XYZ,Normal,Moments,Euclid_Maps] loss multiplication modifiers')
    # Loss Modifiers: # TODO - Implement for Euclid Maps as well.
    p.add_argument('--mask_penalties', nargs=3, type=float, default=[0, 0, 0],
                   help='[XYZ,Normal,Moments] increased weight on mask vertices. Use val <= 1 to disable')
    p.add_argument('--dist_v_penalties', nargs=3, type=float, default=[0, 0, 0],
                   help='[XYZ,Normal,Moments] increased weight on distant vertices. Use val <= 1 to disable')

    # TODO - Assert that if l2_lambda is requried for normals/momenets than input channels are 6,12 etc
    p.add_argument("--stop_num", "-s", type=int, default=100,
                   help="Number of Early Stopping")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Adam's weight decay")

    p = model_cls.add_model_specific_args(p)

    # TODO - double learning rate if you double batch size.

    return p.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Mains
# ----------------------------------------------------------------------------------------------------------------------
def train_main(hparams):
    # Set the random seed:
    set_determinsitic_run()

    model_cls = F2PEncoderDecoder

    # Bring in arguments:
    hparams = parse_args(model_cls)
    # Bring in model:
    nn = model_cls(hparams)

    # Bring in data:
    ds = PointDatasetMenu.get('FaustPyProj')
    my_loaders = ds.split_loaders(split=[0.5, 0.4, 0.1], s_nums=[100, 200, 100],
                                  s_shuffle=[True] * 3, s_transform=[Center()] * 3, batch_size=10)

    # Bridge data and model
    nn.set_loaders(ds)

    # TODO - Handle Resume case :
    #     exp = TestTubeLogger(save_dir=save_dir)
    #     exp.log_hyperparams(args)

    trainer = Trainer()  # Add options
    trainer.fit(nn)
    # The user can try to turn off validation by setting val_check_interval to a big valu
    # # trainer = Trainer(logger=False, - To turn off loging
    # # Remember that checkpoints destroy the checkpoint directory - make sure it is set correctly
    # # assert num examples > batch
    # # from pytorch_lightning

    # trainer.test(model)


def test_main():
    nn = F2PEncoderDecoder.load_from_metrics(
        weights_path='/path/to/pytorch_checkpoint.ckpt',
        tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
        on_gpu=True,
        map_location=None
    )
    ldr = PointDatasetMenu.get('DFaustPyProj').loader(ids=range(1000), transforms=None, batch_size=16)
    # Use only test loader
    nn.set_loaders([None, None, ldr])

    #     early_stop = EarlyStopping(
    #         monitor='avg_val_loss',
    #         patience=args.stop_num,
    #         verbose=False,
    #         mode='min')
    #
    #     checkpoint = ModelCheckpoint(
    #         filepath=save_dir / "checkpoint",
    #         save_best_only=True,
    #         verbose=False,
    #         monitor='avg_val_loss',
    #         mode='min')
    #
    #     trainer = Trainer(
    #         logger=exp,
    #         max_nb_epochs=args.epoch,
    #         checkpoint_callback=checkpoint,
    #         early_stop_callback=early_stop,
    #         gpus=device,
    #         distributed_backend=backend)
    #
    # trainer.test(nn)

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
    # resume_from_checkpoint - Resume from something specific

    #This is the easiest/fastest way which loads hyperparameters
    #and weights from a checkpoint, such as the one saved by the ModelCheckpoint callback
    # pretrained_model = MyLightningModule.load_from_checkpoint(
    #     checkpoint_path='/path/to/pytorch_checkpoint.ckpt'
    # )
    # Train:
    # checkpoint_callback
    # log_gpu_memory = 'min_max' or 'all'
    # track_grad_norm  = 2 # Track L2 norm of the gradient
    # fast_dev_run = True - runs full iteration over everything to find bugs
    # log_save_interval = 100
    # distributed_backend = 'dp', 'ddp', 'ddp2'
    # weights_summary = 'full', 'top' , None
    # accumulate_grad_batches = 1
    # nb_sanity_val_steps = 5 # Sanity checks for validation - Use 0 to optimize
    # if not has_checkpoint:
    #     nb_sanity_val_stes = 3
    #     fast_dev_run = True
    # `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    #           if '.ckpt' in name:
    #                 epoch = name.split('epoch_')[1]
    #                 epoch = int(re.sub('[^0-9]', '', epoch))
    #
    #                 if epoch > last_epoch:
    #                     last_epoch = epoch
    #                     last_ckpt_name = name
    # EarlyStopping(
    #                 monitor='val_loss',
    #                 patience=3,
    #                 verbose=True,
    #                 mode='min'
    #             )
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=os.getcwd(),
    #     save_best_only=True,
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min',
    #     prefix=''
    # )
    # {}_ckpt_epoch_{}.ckpt {prefix}

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Tutorials
# ----------------------------------------------------------------------------------------------------------------------
@tutorial
def dataset_tutorial():
    # Use the menu to see which datasets are implemented
    print(PointDatasetMenu.which())
    ds = PointDatasetMenu.get('FaustPyProj')  # This will fail if you don't have the data on disk
    ds.validate_dataset()  # Make sure all files are available - Only run this once, to make sure.
    banner('The HIT')
    ds.report_index_tree()  # Take a look at how the dataset is indexed - using the hit [HierarchicalIndexTree]

    banner('Collateral Info')
    print(f'Dataset Name = {ds.name()}')
    print(f'Number of available Point Clouds = {ds.num_pnt_clouds()}')
    print(f'Expected Input Shape for a single mesh = {ds.shape()}')
    print(f'Required disk space in bytes = {ds.disk_space()}')
    # You can also request a summary printout with:
    ds.data_summary(with_tree=False)  # Don't print out the tree again
    # For models with a single set of faces (SMPL or SMLR for example) you can request the face set directly:
    banner('Face Tensor')
    print(ds.faces(torch_version=False))
    # You can ask for a random sample of the data, under your needed transformations:
    banner('Data sample')
    print(ds.sample(num_samples=1, transforms=[Center()]))
    # You can also ask for a simple loader, given by the ids you'd like to see.
    # Pass ids = None to index the entire dataset, form point_cloud = 0 to point_cloud = num_point_clouds -1
    my_loader = ds.loader(ids=None, transforms=[Center()], batch_size=16, device='cpu-single')

    # To receive train/validation splits or train/validation/test splits use:
    my_loaders = ds.split_loaders(split=[0.5, 0.4, 0.1], s_nums=[100, 200, 300],
                                  s_shuffle=[True] * 3, s_transform=[Center()] * 3, global_shuffle=True)
    # You'll receive len(split) dataloaders, where each part i is split[i]*num_point_clouds size. From this split,
    # s_nums[i] will be taken for the dataloader, and transformed by s_transform[i].
    # s_shuffle and global_shuffle controls the shuffling of the different partitions - see doc inside function

    # You can see part of the actual dataset using the vtkplotter function, with strategies 'spheres','mesh' or 'cloud'
    ds.show_sample(n_shapes=8, key='gt_part_v', strategy='cloud')


@tutorial
def pytorch_net_tutorial():
    # What is it? PyTorchNet is a derived class of LightningModule, allowing for extended operations on it
    # Let's see some of them:

    nn = F2PEncoderDecoder() # Remember that F2PEncoderDecoder is a subclass of PytorchNet
    nn.identify_system()  # Outputs the specs of the current system - Useful for identifying existing GPUs

    banner('General Net Info')
    print(f'On GPU = {nn.ongpu()}')  # Whether the system is on the GPU or not. Will print False
    target_input_size = ((6890, 3), (6890, 3))
    nn.summary(print_it=True, batch_size=3, x_shape=target_input_size)
    print(f'Output size = {nn.output_size(x_shape=target_input_size)}')

    banner('Weight Print')
    nn.print_weights()
    nn.visualize(x_shape=target_input_size, frmt='pdf')  # Prints PDF to current directory

    # Let's say we have some sort of nn.Module network:
    banner('Some network from the net usecase')
    import torchvision
    nn = torchvision.models.resnet50(pretrained=False)
    # We can extend it's functionality at runtime with a monkeypatch:
    py_nn = PytorchNet.monkeypatch(nn)
    py_nn.summary(x_shape=(3, 28, 28), batch_size=64)


if __name__ == '__main__': pytorch_net_tutorial()
