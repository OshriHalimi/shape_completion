from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import os
import json
import visdom
import time

from dataset import *
from model import *
from utils import *

if __name__ == '__main__':  # OH: Wrapping the main code with __main__ check is necessary for Windows compatibility
    # of the multi-process data loader (see pytorch documentation)
    # =============PARAMETERS======================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=15, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--model', type=str, default=os.path.join(os.getcwd(), "log", "Simple network; Translation augmentation; Input normals; Deeper Decoder", "network_last.pth"), help='optional reload model path')
    parser.add_argument('--save_path', type=str, default='Simple network; Translation augmentation; Input normals; Deeper Decoder', help='save path')
    parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')  #OH: TODO edit

    opt = parser.parse_args()
    print(opt)

    # =============DEFINE stuff for logs ======================================== #
    # Launch visdom for visualization
    vis = visdom.Visdom(port=8888, env=opt.env)
    save_path = opt.save_path
    dir_name = os.path.join('./log/', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = 1 #random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    L2curve_train_smpl = []
    L2curve_val_smlp = []

    # meters to record stats on learning
    train_loss_L2_smpl = AverageValueMeter()
    val_loss_L2_smpl = AverageValueMeter()
    tmp_val_loss = AverageValueMeter()

    # ===================CREATE DATASET================================= #

    dataset = AmassProjectionsDataset(train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    # OH: pin_memory=True used to increase the performance when transferring the fetched data from CPU to GPU
    dataset_test = AmassProjectionsDataset(train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    len_dataset = len(dataset)
    len_dataset_test = len(dataset_test)

    # ===================CREATE network================================= #
    # TODO: update network class AE_AtlasNet_Humans(): The decoder deform a given template, based on the encoding of the template and the partial shape
    network = CompletionNet()
    network.cuda()  # put network on GPU
    network.apply(weights_init)  # initialization of the weight
    if opt.model != '':
        network.load_state_dict(torch.load(opt.model))
        print(" Previous weight loaded ")
    # ========================================================== #

    # ===================CREATE optimizer================================= #
    lrate = 0.001 # learning rate
    optimizer = optim.Adam(network.parameters(), lr=lrate)

    with open(logname, 'a') as f:  # open and append
        f.write(str(network) + '\n')
    # ========================================================== #

    # =============start of the learning loop ======================================== #
    for epoch in range(opt.nepoch):
        if (epoch % 100) == 99:
            lrate = lrate/2.0
            optimizer = optim.Adam(network.parameters(), lr=lrate)

        # TRAIN MODE
        train_loss_L2_smpl.reset()
        network.train()
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            part, template, gt, _ = data

            # OH: place on GPU
            part = part.transpose(2, 1).contiguous().cuda().float()
            template = template.transpose(2, 1).contiguous().cuda().float()
            gt = gt.transpose(2, 1).contiguous().cuda().float()

            # Forward pass
            pointsReconstructed = network(part, template).double()
            #D_reconstructed = calc_euclidean_dist_matrix(pointsReconstructed).double()
            #D_gt = calc_euclidean_dist_matrix(gtV).double()
            #loss_euclidean = torch.mean((D_reconstructed - D_gt) ** 2)

            loss_points = torch.mean((pointsReconstructed[:,:3,:] - gt[:,:3,:].double()) ** 2)

            loss_net = loss_points
            loss_net.backward()
            train_loss_L2_smpl.update(loss_net.item())
            optimizer.step()  # gradient update

            # VIZUALIZE
            if i % 100 == 0:
                # VIZUALIZE
                if i % 100 == 0:
                    vis.scatter(X=part[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Train_Part',
                                opts=dict(title="Train_Part", markersize=2, ), )
                    vis.scatter(X=template[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Train_Template',
                                opts=dict(title="Train_Template", markersize=2, ), )
                    vis.scatter(X=pointsReconstructed[0].transpose(1, 0).contiguous().data.cpu(), win='Train_output',
                                opts=dict(title="Train_output",markersize=2, ), )
                    vis.scatter(X=gt[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Train_Ground_Truth',
                                opts=dict(title="Train_Ground_Truth", markersize=2, ), )

            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / opt.batchSize,  loss_net.item()))

        # Validation
        with torch.no_grad():
            network.eval()
            val_loss_L2_smpl.reset()
            for i, data in enumerate(dataloader_test, 0):
                part, template, gt, _ = data
                # OH: place on GPU
                part = part.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
                template = template.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
                gt = gt.transpose(2, 1).contiguous().cuda(non_blocking=True).float()

                # Forward pass
                pointsReconstructed = network(part, template).double()
                # D_reconstructed = calc_euclidean_dist_matrix(pointsReconstructed).double()
                # D_gt = calc_euclidean_dist_matrix(gtV).double()
                # loss_euclidean = torch.mean((D_reconstructed - D_gt) ** 2)

                loss_points = torch.mean((pointsReconstructed - gt[:,:3,:].double()) ** 2)

                loss_net = loss_points
                val_loss_L2_smpl.update(loss_net.item())

                # VIZUALIZE
                if i % 100 == 0:
                    vis.scatter(X=part[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Test_Part',
                                opts=dict(title="Test_Part", markersize=2, ), )
                    vis.scatter(X=template[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Test_Template',
                                opts=dict(title="Test_Template", markersize=2, ), )
                    vis.scatter(X=pointsReconstructed[0].transpose(1, 0).contiguous().data.cpu(), win='Test_output',
                                opts=dict(title="Test_output",markersize=2, ), )
                    vis.scatter(X=gt[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Test_Ground_Truth',
                                opts=dict(title="Test_Ground_Truth", markersize=2, ), )

                print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset_test / opt.batchSize, loss_net.item()))


            # UPDATE CURVES
            L2curve_train_smpl.append(train_loss_L2_smpl.avg)
            L2curve_val_smlp.append(val_loss_L2_smpl.avg)

            vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smlp)))),
                     Y=np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smlp))),
                     win='loss',
                     opts=dict(title="loss", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smlp" + opt.env,]))

            vis.line(X=np.column_stack((np.arange(len(L2curve_train_smpl)), np.arange(len(L2curve_val_smlp)))),
                     Y=np.log(np.column_stack((np.array(L2curve_train_smpl), np.array(L2curve_val_smlp)))),
                     win='log',
                     opts=dict(title="log", legend=["L2curve_train_smpl" + opt.env,"L2curve_val_smlp" + opt.env,]))

            # dump stats in log file
            log_table = {
                "val_loss_L2_smpl": val_loss_L2_smpl.avg,
                "train_loss_L2_smpl": train_loss_L2_smpl.avg,
                "epoch": epoch,
                "lr": lrate,
                "env": opt.env,
            }
            print(log_table)
            with open(logname, 'a') as f:  # open and append
                f.write('json_stats: ' + json.dumps(log_table) + '\n')
            #save latest network
            torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
