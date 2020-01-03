from argparse import ArgumentParser
import random
import torch
import torch.optim as optim
import visdom
from dataset.datasets import PointDatasetMenu
from util.gen import none_or_int,banner,tutorial
from string import capwords
import numpy as np
import cfg
from test_tube import HyperOptArgumentParser
from architecture.models import CompletionNet
from architecture.PytorchNet import PytorchNet
from dataset.transforms import *
# ----------------------------------------------------------------------------------------------------------------------
#                                                 Tutorials
# ----------------------------------------------------------------------------------------------------------------------
@tutorial
def dataset_tutorial():
    # Use the menu to see which datasets are implemented
    print(PointDatasetMenu.which())
    ds = PointDatasetMenu.get('FaustPyProj') # This will fail if you don't have the data on disk
    ds.validate_dataset() # Make sure all files are available - Only run this once, to make sure.
    banner('The HIT')
    ds.report_index_tree()  # Take a look at how the dataset is indexed - using the hit [HierarchicalIndexTree]

    banner('Collateral Info')
    print(f'Dataset Name = {ds.name()}')
    print(f'Number of available Point Clouds = {ds.num_pnt_clouds()}')
    print(f'Expected Input Shape for a single mesh = {ds.shape()}')
    print(f'Required disk space in bytes = {ds.disk_space()}')
    # You can also request a summary printout with:
    ds.data_summary(with_tree=False) # Don't print out the tree again
    # For models with a single set of faces (SMPL or SMLR for example) you can request the face set directly:
    banner('Face Tensor')
    print(ds.faces(torch_version=False))
    # You can ask for a random sample of the data, under your needed transformations:
    banner('Data sample')
    print(ds.sample(num_samples=1,transforms=[Center()]))
    # You can also ask for a simple loader, given by the ids you'd like to see.
    # Pass ids = None to index the entire dataset, form point_cloud = 0 to point_cloud = num_point_clouds -1
    my_loader = ds.loader(ids=None, transforms=[Center()], batch_size=16, device='cpu-single')

    # To receive train/validation splits or train/validation/test splits use:
    my_loaders = ds.split_loaders(split=[0.5, 0.4, 0.1], s_nums=[100, 200, 300],
                                   s_shuffle=[True]*3,s_transform=[Center()]*3,global_shuffle=True)
    # You'll receive len(split) dataloaders, where each part i is split[i]*num_point_clouds size. From this split,
    # s_nums[i] will be taken for the dataloader, and transformed by s_transform[i].
    # s_shuffle and global_shuffle controls the shuffling of the different partitions - see doc inside function

    # You can see part of the actual dataset using the vtkplotter function, with strategies 'spheres','mesh' or 'cloud'
    ds.show_sample(n_shapes=8,key='gt_part_v',strategy='spheres')


@tutorial
def pytorch_net_tutorial():

    # What is it? PyTorchNet is a derived class of LightningModule, allowing for extended operations on it
    # Let's see some of them:
    nn = CompletionNet() # Use default constructor parameters. Remember that CompNet is a subclass of PytorchNet
    nn.identify_system() # Outputs the specs of the current system - Useful for identifying existing GPUs

    banner('General Net Info')
    print(f'On GPU = {nn.ongpu()}') # Whether the system is on the GPU or not. Will print False
    target_input_size = ((6890,3),(6890,3))
    nn.summary(print_it=True,batch_size=3,x_shape=target_input_size)
    print(f'Output size = {nn.output_size(x_shape=target_input_size)}')

    banner('Weight Print')
    nn.print_weights()
    nn.visualize(x_shape=target_input_size, frmt='pdf') # Prints PDF to current directory

    # Let's say we have some sort of pretrained nn.Module network:
    banner('Pretrained network from the net')
    import torchvision
    nn = torchvision.models.resnet50(False)
    # We can extend it's functionality at runtime with a monkeypatch:
    py_nn = PytorchNet.monkeypatch(nn)
    py_nn.summary(x_shape=(3,28,28),batch_size=64)






#
#
#
# GLOBAL_DS_SHUFFLE = False
# SSHUFFLE = 3
# S_NUMS = 3
# S_TRANSFORM = 3
# # double learning rate if you double batch size.
# # ----------------------------------------------------------------------------------------------------------------------
# #
# # ----------------------------------------------------------------------------------------------------------------------
# # TURN OFF validation - only by erasing the function
# # The user can try to turn off validation by setting val_check_interval to a big valu
# # trainer = Trainer(logger=False, - To turn off loging
# # Remember that checkpoints destroy the checkpoint directory - make sure it is set correctly
# # assert num examples > batch
# # from pytorch_lightning
#
#
#
#
# def parse_cmd_args(p=HyperOptArgumentParser(strategy='random_search')):
#
#     # Exp Config:
#     p.add_argument('--exp_name', type=str, default='General',
#                    help='The experiment name. Leave blank to use the generic exp name')
#
#     # NN Config
#     p.add_argument('--resume_by', choices=['', 'val_acc', 'time'], default='time',
#                    help='Resume Configuration - Either by (1) Latest time OR (2) Best Vald Acc OR (3) No Resume')
#     p.add_argument('--use_visdom', type=bool, default=True)
#
#     # Dataset Config:
#     p.add_argument('--data_samples', nargs=3, type=none_or_int, default=[None,1000,1000],
#                    help='[Train,Validation,Test] number of samples. Use None for all in partition')
#     p.add_argument('--in_channels',choices=[3,6,12],default=6,
#                    help='Number of input channels')
#
#     # Train Config:
#     p.add_argument('--batch_size', type=int, default=16, help='SGD batch size')
#     p.add_argument('--n_epoch', type=int, default=1000, help='The number of epochs to train for')
#
#     # Losses: Use 0 to ignore, >0 to compute
#     p.add_argument('--l2_lambda', nargs=4, type=float, default=[1,0.1,0,0],
#                    help='[XYZ,Normal,Moments,Euclid_Maps] L2 loss multiplication modifiers')
#     # Loss Modifiers: # TODO - Implement for Euclid Maps as well
#     p.add_argument('--l2_mask_penalty',nargs=3,type=float,default=[0,0,0],
#                    help='[XYZ,Normal,Moments] increased weight on mask vertices')
#     p.add_argument('--l2_distant_v_penalty',nargs=3,type=float, default=[0,0,0],
#                    help='[XYZ,Normal,Moments] increased weight on distant vertices')
#     return p.parse_args()
#     parser.add_argument("--stop_num", "-s", type=int, default=100,
#                         help="Number of Early Stopping")
#     parser.add_argument("--weight_decay", type=float, default=1e-4,
#                         help="Adam's weight decay")
#     # Weight decay
#
# def test_only_main():
#     model = CompletionNet.load_from_metrics(
#         weights_path='/path/to/pytorch_checkpoint.ckpt',
#         tags_csv='/path/to/test_tube/experiment/version/meta_tags.csv',
#         on_gpu=True,
#         map_location=None
#     )
#     model.set_loaders()
#
#     # init trainer with whatever options
#     trainer = Trainer(...)
#
#     # test (pass in the model)
#     trainer.test(model)
#
#
# def run(arch, loader, return_test=False):
#     set_random_seed(2434)
#     args = parse_args()
#     device = list(range(args.device))
#     save_dir = make_directory(args.logdir)
#
#     model = MODELS[args.task](arch, loader, args)
#
#     exp = TestTubeLogger(save_dir=save_dir)
#     exp.log_hyperparams(args)
#
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
#     backend = None if len(device) == 1 else "dp"
#
#     trainer = Trainer(
#         logger=exp,
#         max_nb_epochs=args.epoch,
#         checkpoint_callback=checkpoint,
#         early_stop_callback=early_stop,
#         gpus=device,
#         distributed_backend=backend)
#
#     trainer.fit(model)
#     print("##### training finish #####")
#
#     trainer.test(model)
#     print("##### test finish #####")
#
#     if return_test:
#         return model.test_predict
#
#
# def args_to_exp():
#
#     # Launch visdom for visualization
#     if opt.use_visdom:
#         vis = visdom.Visdom(port=8888, env=opt.save_path)
#     save_path = opt.save_path
#     if not os.path.exists("log"):
#         os.mkdir("log")
#     dir_name = os.path.join('./log/', save_path)
#     if not os.path.exists(dir_name):
#         os.mkdir(dir_name)
#     logname = os.path.join(dir_name, 'log.txt')
#     with open(logname, 'a') as f:  # open and append
#         f.write(str(opt) + '\n')
#
#     opt.manualSeed = 1  # random.randint(1, 10000)  # fix seed
#     print("Random Seed: ", opt.manualSeed)
#     random.seed(opt.manualSeed)
#     torch.manual_seed(opt.manualSeed)
#     np.random.seed(opt.manualSeed)
#     Loss_curve_train = []
#     Loss_curve_val = []
#     Loss_curve_val_amass = []
#
#     # meters to record stats on learning
#     train_loss = AverageValueMeter()
#     val_loss = AverageValueMeter()
#     val_loss_amass = AverageValueMeter()
#     tmp_val_loss = AverageValueMeter()
#
#     # ===================CREATE DATASET================================= #
#     dataset = FaustProjectionsPart2PartDataset(train=True, num_input_channels=opt.num_input_channels,
#                                                train_size=opt.p2p_faust_train_size, test_size=opt.p2p_faust_test_size)
#
#     # dataset = AmassProjectionsDataset(split='train', num_input_channels=opt.num_input_channels, filtering=opt.filtering,
#     #                                  mask_penalty=opt.mask_xyz_penalty, use_same_subject=opt.use_same_subject,
#     #                                  train_size=opt.amass_train_size, validation_size=opt.amass_validation_size)
#
#     # dataset = FaustProjectionsDataset(train=True, num_input_channels=opt.num_input_channels,
#     #                                   train_size=opt.faust_train_size, test_size = opt.faust_test_size, mask_penalty=opt.mask_xyz_penalty)
#
#     # dataset = DfaustProjectionsDataset(train=True, num_input_channels=opt.num_input_channels,
#     #                                    train_size=opt.dynamic_faust_train_size, test_size =opt.dynamic_faust_test_size, mask_penalty=opt.mask_xyz_penalty)
#
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
#                                              num_workers=int(opt.workers), pin_memory=True)
#     # OH: pin_memory=True used to increase the performance when transferring the fetched data from CPU to GPU
#     dataset_test = FaustProjectionsDataset(train=False, num_input_channels=opt.num_input_channels,
#                                            train_size=opt.faust_train_size, mask_penalty=opt.mask_xyz_penalty)
#
#     dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True,
#                                                   num_workers=int(opt.workers), pin_memory=True)
#     dataset_test_amass = AmassProjectionsDataset(split='validation', num_input_channels=opt.num_input_channels,
#                                                  filtering=opt.filtering,
#                                                  mask_penalty=opt.mask_xyz_penalty,
#                                                  use_same_subject=opt.use_same_subject,
#                                                  train_size=opt.amass_train_size,
#                                                  validation_size=opt.amass_validation_size)
#     dataloader_test_amass = torch.utils.data.DataLoader(dataset_test_amass, batch_size=opt.batchSize, shuffle=True,
#                                                         num_workers=int(opt.workers), pin_memory=True)
#
#     len_dataset = len(dataset)
#     len_dataset_test = len(dataset_test)
#     len_dataset_test_amass = len(dataset_test_amass)
#
#     # get dataset triangulations
#     if opt.normal_loss_slope > 0:
#         triv_tup = dataset_test.triangulation(use_torch=True)
#     else:
#         triv_tup = None
#
#     # ===================CREATE network================================= #
#     network = CompletionNet(num_input_channels=opt.num_input_channels, num_output_channels=opt.num_output_channels,
#                             centering=opt.centering)
#     network.cuda()  # put network on GPU
#     network.apply(weights_init)  # initialization of the weight
#     old_epoch = 0
#     try:
#         if opt.model_dir != '':
#             model_path = os.path.join(os.getcwd(), "log", opt.model_dir, opt.model_file)
#             print(model_path)
#             network.load_state_dict(torch.load(model_path))
#             print(" Previous weight loaded ")
#             log_path = os.path.join(os.getcwd(), "log", opt.model_dir, "log.txt")
#             print(log_path)
#             old_epoch = read_lr(log_path)
#     except:
#         print('Saved weights mismatch in input size - Retraining')
#     # ========================================================== #
#
#     # ===================CREATE optimizer================================= #
#     lrate = 0.001 / max(1, (old_epoch // 100) * 2)  # learning rate
#     optimizer = optim.Adam(network.parameters(), lr=lrate)
#
#     with open(logname, 'a') as f:  # open and append
#         f.write(str(network) + '\n')
#     # ========================================================== #
#
#     # =============start of the learning loop ======================================== #
#     for epoch in range(opt.nepoch):
#         part = None
#         template = None
#         pointsReconstructed = None
#         gt = None
#         mask = None
#
#         if (epoch % 100) == 99:
#             lrate = lrate / 2.0
#             optimizer = optim.Adam(network.parameters(), lr=lrate)
#
#         # TRAIN MODE
#         train_loss.reset()
#         network.train()
#         for i, data in enumerate(dataloader, 0):
#
#             optimizer.zero_grad()
#
#             # Bring in data to GPU
#             # template = Full shape, uses part to achieve the gt
#             # part = gt[mask_full] - A part of the gt, used to direct template where to advance
#             # gt = The ground truth - Our optimization target
#             # mask = A vector of ones and mask_penalty values, where mask[:,:,[i]] == mask_penalty for vertices in part
#
#             if dataset.__class__.__name__ == 'FaustProjectionsDataset':
#                 part, template, gt, subject_id_full, mask_loss = data
#             else:
#                 template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss, _ = data
#
#             part = part.transpose(2, 1).contiguous().cuda().float()
#             template = template.transpose(2, 1).contiguous().cuda().float()
#             gt = gt.transpose(2, 1).contiguous().cuda().float()
#             mask_loss = torch.unsqueeze(mask_loss, 2).transpose(2, 1).contiguous().cuda().float()  # [B x 1 x N]
#             # Forward pass
#             gt_rec, _, part_centering = network(part, template)
#             # Center gt
#             gt[:, :3, :] -= part_centering
#
#             loss = compute_loss(gt, gt_rec, template, mask_loss, triv_tup, opt)
#
#             train_loss.update(loss.item())
#             loss.backward()
#             optimizer.step()  # gradient update
#
#             # VIZUALIZE
#             if opt.use_visdom and i % 100 == 0:
#                 vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Part',
#                             opts=dict(title="Train_Part", markersize=2, ), )
#                 vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Template',
#                             opts=dict(title="Train_Template", markersize=2, ), )
#                 vis.scatter(X=gt_rec[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_output',
#                             opts=dict(title="Train_output", markersize=2, ), )
#                 vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Ground_Truth',
#                             opts=dict(title="Train_Ground_Truth", markersize=2, ), )
#
#             print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / opt.batchSize, loss.item()))
#
#         # Validation Faust
#         with torch.no_grad():
#             network.eval()
#             val_loss.reset()
#             for i, data in enumerate(dataloader_test, 0):
#                 part, template, gt, subject_id_full, mask_loss = data
#                 # OH: place on GPU
#                 part = part.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
#                 template = template.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
#                 gt = gt.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
#                 mask_loss = torch.unsqueeze(mask_loss, 2).transpose(2, 1).contiguous().cuda().float()
#                 # Forward pass
#                 gt_rec, _, part_centering = network(part, template)
#                 gt[:, :3, :] -= part_centering
#
#                 # In Faust validation we don't use the mask right now (Faust dataloader doesn't return the mask yet)
#                 # TODO: return the indices of the part of the part within Faust dataloader
#                 loss = compute_loss(gt, gt_rec, template, mask_loss, triv_tup, opt)
#                 val_loss.update(loss.item())
#
#                 # VIZUALIZE
#                 if opt.use_visdom and i % 100 == 0:
#                     vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Part',
#                                 opts=dict(title="Test_Part", markersize=2, ), )
#                     vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Template',
#                                 opts=dict(title="Test_Template", markersize=2, ), )
#                     vis.scatter(X=gt_rec[0].transpose(1, 0).contiguous().data.cpu(), win='Test_output',
#                                 opts=dict(title="Test_output", markersize=2, ), )
#                     vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Ground_Truth',
#                                 opts=dict(title="Test_Ground_Truth", markersize=2, ), )
#
#                 print('[%d: %d/%d] test loss:  %f' % (epoch, i, len_dataset_test / opt.batchSize, loss.item()))
#
#         # Validation AMASS
#         with torch.no_grad():
#             network.eval()
#             val_loss_amass.reset()
#             for i, data in enumerate(dataloader_test_amass, 0):
#                 template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss, _ = data
#                 # OH: place on GPU
#                 part = part.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
#                 template = template.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
#                 gt = gt.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
#                 mask_loss = torch.unsqueeze(mask_loss, 2).transpose(2, 1).contiguous().cuda().float()  # [B x 1 x N]
#
#                 # Forward pass
#                 gt_rec, _, part_centering = network(part, template)
#                 gt[:, :3, :] -= part_centering
#                 loss = compute_loss(gt, gt_rec, template, None, triv_tup, opt)
#
#                 val_loss_amass.update(loss.item())
#
#                 # VIZUALIZE
#                 if opt.use_visdom and i % 100 == 0:
#                     vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Amass_Part',
#                                 opts=dict(title="Test_Amass_Part", markersize=2, ), )
#                     vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Amass_Template',
#                                 opts=dict(title="Test_Amass_Template", markersize=2, ), )
#                     vis.scatter(X=gt_rec[0, :3, :].transpose(1, 0).contiguous().data.cpu(),
#                                 win='Test_Amass_output',
#
#                                 opts=dict(title="Test_Amass_output", markersize=2, ), )
#                     vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Amass_Ground_Truth',
#                                 opts=dict(title="Test_Amass_Ground_Truth", markersize=2, ), )
#
#                 print(
#                     '[%d: %d/%d] test loss:  %f' % (epoch, i, len_dataset_test_amass / opt.batchSize, loss.item()))
#
#         # UPDATE CURVES
#         Loss_curve_train.append(train_loss.avg)
#         Loss_curve_val.append(val_loss.avg)
#         Loss_curve_val_amass.append(val_loss_amass.avg)
#
#         if opt.use_visdom:
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
#                      Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val))),
#                      win='Faust loss',
#                      opts=dict(title="Faust loss", legend=["Train loss", "Faust Validation loss", ]))
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
#                      Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val)))),
#                      win='"Faust log loss',
#                      opts=dict(title="Faust log loss", legend=["Train loss", "Faust Validation loss", ]))
#
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
#                      Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass))),
#                      win='AMASS loss',
#                      opts=dict(title="AMASS loss", legend=["Train loss", "Validation loss", "Validation loss amass"]))
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
#                      Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass)))),
#                      win='AMASS log loss',
#                      opts=dict(title="AMASS log loss", legend=["Train loss", "Faust Validation loss", ]))
#
#         # dump stats in log file
#         log_table = {
#             "val_loss": val_loss.avg,
#             "train_loss": train_loss.avg,
#             "val_loss_amass": val_loss_amass.avg,
#             "epoch": epoch,
#             "lr": lrate,
#             "env": opt.save_path,
#         }
#         print(log_table)
#         with open(logname, 'a') as f:  # open and append
#             f.write('json_stats: ' + json.dumps(log_table) + '\n')
#             f.write("EPOCH NUMBER: " + str(epoch + old_epoch) + "\n")
#
#         # save latest network
#         torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
#
#
if __name__ == '__main__': dataset_tutorial()
