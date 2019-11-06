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
import matplotlib.pyplot as plt
import scipy.io as sio

from dataset import *
from model import *
from utils import *

try:
    import visdom
except:
    print("Please install the module 'visdom' for visualization, e.g.")
    print("pip install visdom")
    sys.exit(-1)

if __name__ == '__main__':  # OH: Wrapping the main code with __main__ check is necessary for Windows compatibility
    # of the multi-process data loader (see pytorch documentation)
    # =============PARAMETERS======================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=15, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--model', type=str, default='',help='optional reload model path')
    parser.add_argument('--save_path', type=str, default='experiment with AMASS data (incomplete)', help='save path')
    parser.add_argument('--env', type=str, default="shape_completion", help='visdom environment')  # OH: TODO edit
    parser.add_argument('--saveOffline', type=bool, default=False)

    opt = parser.parse_args()
    print(opt)
    ts = 0
    if opt.saveOffline:
        ts = time.time()
        ts = int(ts)
        os.mkdir(ts)

    # =============DEFINE stuff for logs ======================================== #
    # Launch visdom for visualization
    vis = visdom.Visdom(port=8888, env=opt.env)
    save_path = opt.save_path
    dir_name = os.path.join('./log/', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')

    opt.manualSeed = 1  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    Loss_curve_train = []
    Loss_curve_val = []

    # meters to record stats on learning
    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()
    tmp_val_loss = AverageValueMeter()

    # ===================CREATE DATASET================================= #

    dataset = AmassProjectionsDataset(train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                             num_workers=int(opt.workers), pin_memory=True)
    # OH: pin_memory=True used to increase the performance when transferring the fetched data from CPU to GPU
    dataset_test = AmassProjectionsDataset(train=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True,
                                                  num_workers=int(opt.workers), pin_memory=True)
    len_dataset = len(dataset)
    len_dataset_test = len(dataset_test)

    # ===================CREATE network================================= #
    network = CompletionNet()
    network.cuda()  # put network on GPU
    network.apply(weights_init)  # initialization of the weight
    if opt.model != '':
        network.load_state_dict(torch.load(opt.model))
        print(" Previous weight loaded ")
    # ========================================================== #

    # ===================CREATE optimizer================================= #
    lrate = 0.001  # learning rate
    optimizer = optim.Adam(network.parameters(), lr=lrate)

    with open(logname, 'a') as f:  # open and append
        f.write(str(network) + '\n')
    # ========================================================== #

    # =============start of the learning loop ======================================== #
    for epoch in range(opt.nepoch):
        part = None
        template = None
        pointsReconstructed = None
        gt = None

        if (epoch % 100) == 99:
            lrate = lrate / 2.0
            optimizer = optim.Adam(network.parameters(), lr=lrate)

        # TRAIN MODE
        train_loss.reset()
        network.train()
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            part, template, gt, _ = data

            # OH: place on GPU
            part = part.transpose(2, 1).contiguous().cuda().float()
            template = template.transpose(2, 1).contiguous().cuda().float()
            gt = gt.transpose(2, 1).contiguous().cuda().float()

            # Forward pass
            pointsReconstructed = network(part, template)
            loss_points = torch.mean((pointsReconstructed[:, :3, :] - gt[:, :3, :]) ** 2)

            loss_net = loss_points
            loss_net.backward()
            train_loss.update(loss_net.item())
            optimizer.step()  # gradient update

            # VIZUALIZE
            if i % 100 == 0:
                vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Part',
                            opts=dict(title="Train_Part", markersize=2, ), )
                vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Template',
                            opts=dict(title="Train_Template", markersize=2, ), )
                vis.scatter(X=pointsReconstructed[0].transpose(1, 0).contiguous().data.cpu(), win='Train_output',
                            opts=dict(title="Train_output", markersize=2, ), )
                vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Ground_Truth',
                            opts=dict(title="Train_Ground_Truth", markersize=2, ), )

            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / opt.batchSize, loss_net.item()))

        if opt.saveOffline: #save the last training batch in each epoch
            for i in range(opt.batchSize):
                tmp_fig = plt.figure()
                ax_train_part = tmp_fig.add_subplot(141, projection='3d')
                ax_train_template = tmp_fig.add_subplot(142, projection='3d')
                ax_train_output = tmp_fig.add_subplot(143, projection='3d')
                ax_train_ground_truth = tmp_fig.add_subplot(144, projection='3d')
                ax_train_part.scatter(part[i, 0, :].contiguous().data.cpu(),
                                      part[i, 1, :].contiguous().data.cpu(),
                                      part[i, 2, :].contiguous().data.cpu())
                ax_train_template.scatter(template[i, 0, :].contiguous().data.cpu(),
                                      template[i, 1, :].contiguous().data.cpu(),
                                      template[i, 2, :].contiguous().data.cpu())
                ax_train_output.scatter(pointsReconstructed[i, 0, :].contiguous().data.cpu(),
                                          pointsReconstructed[i, 1, :].contiguous().data.cpu(),
                                          pointsReconstructed[i, 2, :].contiguous().data.cpu())
                ax_train_ground_truth.scatter(gt[i, 0, :].contiguous().data.cpu(),
                                          gt[i, 1, :].contiguous().data.cpu(),
                                          gt[i, 2, :].contiguous().data.cpu())
                plt.savefig(os.path.join(str(ts),'train_'+str(epoch)+'_'+str(i)+'.png'))

            sio.savemat('train_'+str(epoch)+'.mat',{"Train_Part":part[:, :3, :].contiguous().data.cpu(),
                                       "Train_Template":template[:, :3, :].contiguous().data.cpu(),
                                       "Train_output":pointsReconstructed[0, :3, :].contiguous().data.cpu(),
                                       "Train_Ground_Truth":gt[:, :3, :].contiguous().data.cpu()})


        # Validation
        with torch.no_grad():
            network.eval()
            val_loss.reset()
            for i, data in enumerate(dataloader_test, 0):
                part, template, gt, _ = data
                # OH: place on GPU
                part = part.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
                template = template.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
                gt = gt.transpose(2, 1).contiguous().cuda(non_blocking=True).float()

                # Forward pass
                pointsReconstructed = network(part, template).double()
                loss_points = torch.mean((pointsReconstructed - gt[:, :3, :].double()) ** 2)

                loss_net = loss_points
                val_loss.update(loss_net.item())

                # VIZUALIZE
                if i % 100 == 0:
                    vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Part',
                                opts=dict(title="Test_Part", markersize=2, ), )
                    vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Template',
                                opts=dict(title="Test_Template", markersize=2, ), )
                    vis.scatter(X=pointsReconstructed[0].transpose(1, 0).contiguous().data.cpu(), win='Test_output',
                                opts=dict(title="Test_output", markersize=2, ), )
                    vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Ground_Truth',
                                opts=dict(title="Test_Ground_Truth", markersize=2, ), )

                print('[%d: %d/%d] test smlp loss:  %f' % (epoch, i, len_dataset_test / opt.batchSize, loss_net.item()))

            if opt.saveOffline: #save the last validation batch in each epoch
                for i in range(opt.batchSize):
                    tmp_fig = plt.figure()
                    ax_test_part = tmp_fig.add_subplot(141, projection='3d')
                    ax_test_template = tmp_fig.add_subplot(142, projection='3d')
                    ax_test_output = tmp_fig.add_subplot(143, projection='3d')
                    ax_test_ground_truth = tmp_fig.add_subplot(144, projection='3d')
                    ax_test_part.scatter(part[i, 0, :].contiguous().data.cpu(),
                                          part[i, 1, :].contiguous().data.cpu(),
                                          part[i, 2, :].contiguous().data.cpu())
                    ax_test_template.scatter(template[i, 0, :].contiguous().data.cpu(),
                                              template[i, 1, :].contiguous().data.cpu(),
                                              template[i, 2, :].contiguous().data.cpu())
                    ax_test_output.scatter(pointsReconstructed[i, 0, :].contiguous().data.cpu(),
                                            pointsReconstructed[i, 1, :].contiguous().data.cpu(),
                                            pointsReconstructed[i, 2, :].contiguous().data.cpu())
                    ax_test_ground_truth.scatter(gt[i, 0, :].contiguous().data.cpu(),
                                                  gt[i, 1, :].contiguous().data.cpu(),
                                                  gt[i, 2, :].contiguous().data.cpu())
                    plt.savefig(os.path.join(str(ts), 'test_' + str(epoch) + '_' + str(i) + '.png'))

                sio.savemat('test_' + str(epoch) + '.mat', {"Test_Part": part[:, :3, :].contiguous().data.cpu(),
                                                             "Test_Template": template[:, :3,
                                                                               :].contiguous().data.cpu(),
                                                             "Test_output": pointsReconstructed[0, :3,
                                                                             :].contiguous().data.cpu(),
                                                             "Test_Ground_Truth": gt[:, :3,
                                                                                   :].contiguous().data.cpu()})
            # UPDATE CURVES
            Loss_curve_train.append(train_loss.avg)
            Loss_curve_val.append(val_loss.avg)

            vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
                     Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val))),
                     win='loss',
                     opts=dict(title="loss", legend=["Train loss", "Validation loss", ]))

            vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
                     Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val)))),
                     win='log',
                     opts=dict(title="log", legend=["Train loss", "Validation loss", ]))

            # dump stats in log file
            log_table = {
                "val_loss": val_loss.avg,
                "train_loss": train_loss.avg,
                "epoch": epoch,
                "lr": lrate,
                "env": opt.env,
            }
            print(log_table)
            with open(logname, 'a') as f:  # open and append
                f.write('json_stats: ' + json.dumps(log_table) + '\n')
            # save latest network
            torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
