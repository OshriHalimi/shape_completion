from __future__ import print_function
import argparse
import random
# import numpy as np
import torch
import torch.optim as optim
import os
import json
import visdom
import time
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from dataset import *
from model import *
from utils import *
import visdom

if __name__ == '__main__':  # OH: Wrapping the main code with __main__ check is necessary for Windows compatibility
    # of the multi-process data loader (see pytorch documentation)
    # =============PARAMETERS======================================== #
    parser = argparse.ArgumentParser()
    # Learning params
    parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')

    # folder where to take pre-trained parameters
    parser.add_argument('--model_dir', type=str, default='',
                        help='optional reload model directory')
    parser.add_argument('--model_file', type=str, default='',
                        help='optional reload model file in model directory')
    # folder that stores the log for the run
    parser.add_argument('--save_path', type=str, default='ID007_normal_loss_01', help='save path')
    parser.add_argument('--env', type=str, default="shape_completion", help='visdom environment')  # OH: TODO edit

    parser.add_argument('--saveOffline', type=bool, default=False)

    # Network params
    parser.add_argument('--num_input_channels', type=int, default=6)
    parser.add_argument('--num_output_channels', type=int,
                        default=3)  # We assume the network return predicted xyz as 3 channels
    parser.add_argument('--use_same_subject', type=bool, default=True)
    # OH: a flag wether to use the same subject in AMASS examples (or two different subjects)
    parser.add_argument('--centering', type=bool, default=True)
    # OH: indicating whether the shapes are centerd w.r.t center of mass before entering the network

    # Dataset params
    parser.add_argument('--amass_train_size', type=int, default=5)
    parser.add_argument('--amass_validation_size', type=int, default=10000)
    parser.add_argument('--amass_test_size', type=int, default=200)
    parser.add_argument('--faust_train_size', type=int, default=10000)
    parser.add_argument('--filtering', type=float, default=0.09, help='amount of filtering to apply on l2 distances')

    # Loss params
    parser.add_argument('--loss_use_normals', type=bool, default=True,
                        help='flag: L2 loss is calculated with normals of the reconstructed shapes: [Yes/Ne]')  # Warning: don't use this feature yet
    parser.add_argument('--loss_normals_weight', type=float, default=0.01,
                        help='considered only if loss_use_normals == True. weight of normals in L2 loss')  # Warning: don't use this feature yet
    parser.add_argument('--penalty_loss', type=float, default=1, help='penalty applied to points belonging to the mask')
    parser.add_argument('--apply_penalty_on_normals', type=bool, default=False,
                        help='flag: considered only if loss_use_normals == True. Do we apply mask penalty also on normals: [Yes/No]')  # Warning: don't use this feature yet

    parser.add_argument('--use_visdom', type=bool, default=False)

    opt = parser.parse_args()
    print(opt)
    ts = 0
    if opt.saveOffline:
        ts = time.time()
        ts = int(ts)
        os.mkdir(ts)

    # =============DEFINE stuff for logs ======================================== #
    # Launch visdom for visualization
    if opt.use_visdom:
        vis = visdom.Visdom(port=8888, env=opt.env)
    save_path = opt.save_path
    if not os.path.exists("log"):
        os.mkdir("log")
    dir_name = os.path.join('./log/', save_path)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    logname = os.path.join(dir_name, 'log.txt')
    with open(logname, 'a') as f:  # open and append
        f.write(str(opt) + '\n')

    opt.manualSeed = 1  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    Loss_curve_train = []
    Loss_curve_val = []
    Loss_curve_val_amass = []

    # meters to record stats on learning
    train_loss = AverageValueMeter()
    val_loss = AverageValueMeter()
    val_loss_amass = AverageValueMeter()
    tmp_val_loss = AverageValueMeter()

    # ===================CREATE DATASET================================= #

    dataset = AmassProjectionsDataset(split='train', num_input_channels=opt.num_input_channels, filtering=opt.filtering,
                                      mask_penalty=opt.penalty_loss, use_same_subject=opt.use_same_subject,
                                      train_size=opt.amass_train_size, validation_size=opt.amass_validation_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                             num_workers=int(opt.workers), pin_memory=True)
    # OH: pin_memory=True used to increase the performance when transferring the fetched data from CPU to GPU
    dataset_test = FaustProjectionsDataset(train=True, num_input_channels=opt.num_input_channels,
                                           train_size=opt.faust_train_size)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True,
                                                  num_workers=int(opt.workers), pin_memory=True)
    dataset_test_amass = AmassProjectionsDataset(split='validation', num_input_channels=opt.num_input_channels,
                                                 filtering=opt.filtering,
                                                 mask_penalty=opt.penalty_loss, use_same_subject=opt.use_same_subject,
                                                 train_size=opt.amass_train_size,
                                                 validation_size=opt.amass_validation_size)
    dataloader_test_amass = torch.utils.data.DataLoader(dataset_test_amass, batch_size=opt.batchSize, shuffle=True,
                                                        num_workers=int(opt.workers), pin_memory=True)

    len_dataset = len(dataset)
    len_dataset_test = len(dataset_test)
    len_dataset_test_amass = len(dataset_test_amass)

    # get dataset triangulations
    if opt.loss_use_normals:
        dataset_triv = dataset.triangulation()
        dataset_test_amass = dataset_test_amass.triangulation()

    # ===================CREATE network================================= #
    network = CompletionNet(num_input_channels=opt.num_input_channels, num_output_channels=opt.num_output_channels,
                            centering=opt.centering)
    network.cuda()  # put network on GPU
    network.apply(weights_init)  # initialization of the weight
    old_epoch = 0
    try:
        if opt.model_dir != '':
            model_path = os.path.join(os.getcwd(), "log", opt.model_dir, opt.model_file)
            print(model_path)
            network.load_state_dict(torch.load(model_path))
            print(" Previous weight loaded ")
            log_path = os.path.join(os.getcwd(), "log", opt.model_dir, "log.txt")
            print(log_path)
            old_epoch = read_lr(log_path)
    except:
        print('Saved weights mismatch in input size - Retraining')
    # ========================================================== #

    # ===================CREATE optimizer================================= #
    lrate = 0.001 / max(1, (old_epoch // 100) * 2)  # learning rate
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
        mask = None

        if (epoch % 100) == 99:
            lrate = lrate / 2.0
            optimizer = optim.Adam(network.parameters(), lr=lrate)

        # TRAIN MODE
        train_loss.reset()
        network.train()
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss, _ = data

            # OH: place on GPU
            part = part.transpose(2, 1).contiguous().cuda().float()
            template = template.transpose(2, 1).contiguous().cuda().float()
            gt = gt.transpose(2, 1).contiguous().cuda().float()

            # Forward pass
            pointsReconstructed, shift_template, shift_part = network(part, template)
            gt[:, :3, :] = gt[:, :3, :] - shift_part

            mask = torch.unsqueeze(mask_loss, 2).transpose(2, 1).contiguous().cuda().float()  # [B x 1 x N]
            loss_points = torch.mean(mask * ((pointsReconstructed[:, :3, :] - gt[:, :3, :]) ** 2))

            if opt.loss_use_normals:
                pointsReconstructed_n = compute_vertex_normals_batch(pointsReconstructed[:, :3, :], dataset_triv)
                gt_n = compute_vertex_normals_batch(gt[:, :3, :], dataset_triv)
                if opt.apply_penalty_on_normals:
                    loss_normals = torch.mean(mask * ((pointsReconstructed_n - gt_n) ** 2))
                else:
                    loss_normals = torch.mean((pointsReconstructed_n - gt_n) ** 2)
            else:
                loss_normals = 0

            loss_net = loss_points + opt.loss_normals_weight * loss_normals
            train_loss.update(loss_net.item())
            loss_net.backward()
            optimizer.step()  # gradient update

            # VIZUALIZE
            if opt.use_visdom and i % 100 == 0:
                vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Part',
                            opts=dict(title="Train_Part", markersize=2, ), )
                vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Template',
                            opts=dict(title="Train_Template", markersize=2, ), )
                vis.scatter(X=pointsReconstructed[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_output',
                            opts=dict(title="Train_output", markersize=2, ), )
                vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Ground_Truth',
                            opts=dict(title="Train_Ground_Truth", markersize=2, ), )

            print('[%d: %d/%d] train loss:  %f' % (epoch, i, len_dataset / opt.batchSize, loss_net.item()))

        # Validation Faust
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
                pointsReconstructed, shift_template, shift_part = network(part, template)
                gt[:, :3, :] = gt[:, :3, :] - shift_part

                # In Faust validation we don't use the mask right now (Faust dataloader doesn't return the mask yet)
                # TODO: return the indices of the part of the part within Faust dataloader
                loss_points = torch.mean((pointsReconstructed[:, :3, :] - gt[:, :3, :]) ** 2)
                if opt.loss_use_normals:
                    pointsReconstructed_n = compute_vertex_normals_batch(pointsReconstructed[:, :3, :],
                                                                         dataset_test_amass)
                    gt_n = compute_vertex_normals_batch(gt[:, :3, :], dataset_test_amass)
                    loss_normals = torch.mean((pointsReconstructed_n - gt_n) ** 2)
                else:
                    loss_normals = 0

                loss_net = loss_points + opt.loss_normals_weight * loss_normals
                val_loss.update(loss_net.item())

                # VIZUALIZE
                if opt.use_visdom and i % 100 == 0:
                    vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Part',
                                opts=dict(title="Test_Part", markersize=2, ), )
                    vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Template',
                                opts=dict(title="Test_Template", markersize=2, ), )
                    vis.scatter(X=pointsReconstructed[0].transpose(1, 0).contiguous().data.cpu(), win='Test_output',
                                opts=dict(title="Test_output", markersize=2, ), )
                    vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Ground_Truth',
                                opts=dict(title="Test_Ground_Truth", markersize=2, ), )

                print('[%d: %d/%d] test loss:  %f' % (epoch, i, len_dataset_test / opt.batchSize, loss_net.item()))

        # Validation AMASS
        with torch.no_grad():
            network.eval()
            val_loss_amass.reset()
            for i, data in enumerate(dataloader_test_amass, 0):
                template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss, _ = data
                # OH: place on GPU
                part = part.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
                template = template.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
                gt = gt.transpose(2, 1).contiguous().cuda(non_blocking=True).float()

                # Forward pass
                pointsReconstructed, shift_template, shift_part = network(part, template)
                gt[:, :3, :] = gt[:, :3, :] - shift_part

                mask = torch.unsqueeze(mask_loss, 2).transpose(2, 1).contiguous().cuda().float()  # [B x 1 x N]
                loss_points = torch.mean(mask * ((pointsReconstructed[:, :3, :] - gt[:, :3, :]) ** 2))

                if opt.loss_use_normals:
                    pointsReconstructed_n = compute_vertex_normals_batch(pointsReconstructed[:, :3, :], dataset_triv)
                    gt_n = compute_vertex_normals_batch(gt[:, :3, :], dataset_triv)
                    if opt.apply_penalty_on_normals:
                        loss_normals = torch.mean(mask * ((pointsReconstructed_n - gt_n) ** 2))
                    else:
                        loss_normals = torch.mean((pointsReconstructed_n - gt_n) ** 2)
                else:
                    loss_normals = 0

                loss_net = loss_points + opt.loss_normals_weight * loss_normals

                val_loss_amass.update(loss_net.item())

                # VIZUALIZE
                if opt.use_visdom and i % 100 == 0:
                    vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Amass_Part',
                                opts=dict(title="Test_Amass_Part", markersize=2, ), )
                    vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Amass_Template',
                                opts=dict(title="Test_Amass_Template", markersize=2, ), )
                    vis.scatter(X=pointsReconstructed[0, :3, :].transpose(1, 0).contiguous().data.cpu(),
                                win='Test_Amass_output',

                                opts=dict(title="Test_Amass_output", markersize=2, ), )
                    vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Test_Amass_Ground_Truth',
                                opts=dict(title="Test_Amass_Ground_Truth", markersize=2, ), )

                print(
                    '[%d: %d/%d] test loss:  %f' % (epoch, i, len_dataset_test_amass / opt.batchSize, loss_net.item()))

        # UPDATE CURVES
        Loss_curve_train.append(train_loss.avg)
        Loss_curve_val.append(val_loss.avg)
        Loss_curve_val_amass.append(val_loss_amass.avg)

        if opt.use_visdom:
            vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
                     Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val))),
                     win='Faust loss',
                     opts=dict(title="Faust loss", legend=["Train loss", "Faust Validation loss", ]))
            vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
                     Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val)))),
                     win='"Faust log loss',
                     opts=dict(title="Faust log loss", legend=["Train loss", "Faust Validation loss", ]))

            vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
                     Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass))),
                     win='AMASS loss',
                     opts=dict(title="AMASS loss", legend=["Train loss", "Validation loss", "Validation loss amass"]))
            vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
                     Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass)))),
                     win='AMASS log loss',
                     opts=dict(title="AMASS log loss", legend=["Train loss", "Faust Validation loss", ]))

        # dump stats in log file
        log_table = {
            "val_loss": val_loss.avg,
            "train_loss": train_loss.avg,
            "val_loss_amass": val_loss_amass.avg,
            "epoch": epoch,
            "lr": lrate,
            "env": opt.env,
        }
        print(log_table)
        with open(logname, 'a') as f:  # open and append
            f.write('json_stats: ' + json.dumps(log_table) + '\n')
            f.write("EPOCH NUMBER: " + str(epoch + old_epoch) + "\n")

        # save latest network
        torch.save(network.state_dict(), '%s/network_last.pth' % (dir_name))
