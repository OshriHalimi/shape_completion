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
import scipy.io as sio

if __name__ == '__main__':  # OH: Wrapping the main code with __main__ check is necessary for Windows compatibility
    # of the multi-process data loader (see pytorch documentation)
    # =============PARAMETERS======================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--model', type=str, default='D:\shape_completion\python code\log/Simple network; Translation augmentation; Input normals; Deeper Decoder/network_last.pth', help='optional reload model path')
    parser.add_argument('--save_path', type=str, default='D:\shape_completion\python code\log/Simple network; Translation augmentation; Input normals; Deeper Decoder/test/', help='save path')
    parser.add_argument('--env', type=str, default="3DCODED_supervised", help='visdom environment')  #OH: TODO edit

    opt = parser.parse_args()
    print(opt)

    # =============DEFINE stuff for logs ======================================== #
    # Launch visdom for visualization
    vis = visdom.Visdom(port=8888, env=opt.env)
    save_path = opt.save_path

    blue = lambda x: '\033[94m' + x + '\033[0m'

    # ===================CREATE DATASET================================= #
    # OH: pin_memory=True used to increase the performance when transferring the fetched data from CPU to GPU
    dataset_test = SHREC16CutsDavidDataset()
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    len_dataset_test = len(dataset_test)

    # ===================CREATE network================================= #
    network = CompletionNet()
    network.cuda()  # put network on GPU
    network.apply(weights_init)  # initialization of the weight
    if opt.model != '':
        network.load_state_dict(torch.load(opt.model))
        print(" Previous weight loaded ")
    # ========================================================== #


    # Test
    with torch.no_grad():
        network.eval()

        for i, data in enumerate(dataloader_test, 0):
            part, template, name, _ = data
            # OH: place on GPU
            part = part.transpose(2, 1).contiguous().cuda(non_blocking=True).float()
            template = template.transpose(2, 1).contiguous().cuda(non_blocking=True).float()

            # Forward pass
            pointsReconstructed = network(part, template).double()
            sio.savemat(save_path + name[0] + '.mat', {'pointsReconstructed': pointsReconstructed.cpu().numpy()})
            # VIZUALIZE
            vis.scatter(X=part[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Test_Part',
                        opts=dict(title="Test_Part", markersize=2, ), )
            vis.scatter(X=template[0,:3,:].transpose(1, 0).contiguous().data.cpu(), win='Test_Template',
                        opts=dict(title="Test_Template", markersize=2, ), )
            vis.scatter(X=pointsReconstructed[0].transpose(1, 0).contiguous().data.cpu(), win='Test_output',
                        opts=dict(title="Test_output",markersize=2, ), )


