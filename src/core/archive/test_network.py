from __future__ import print_function
import argparse
import random
#import numpy as np

from archive.old_dataset import *
from model import *

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
    # Learning params
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)

    # Path params
    parser.add_argument('--model_dir', type=str, default='experiment with AMASS data (incomplete)',help='optional reload model directory')
    parser.add_argument('--model_file', type=str, default='network_last.pth',help='optional reload model file in model directory')
    parser.add_argument('--save_path', type=str, default='Amass test set generalization', help='save path')
    parser.add_argument('--env', type=str, default="shape_completion", help='visdom environment')  # OH: TODO edit
    parser.add_argument('--saveOffline', type=bool, default=False)

    #Network params
    parser.add_argument('--num_input_channels', type=int, default=3)
    parser.add_argument('--use_same_subject', type=bool, default=True) #OH: a flag wether to use the same subject in AMASS examples (or two different subjects)
    parser.add_argument('--centering', type=bool, default=True) #OH: indicating whether the shapes are centerd w.r.t center of mass before entering the network

    #Dataset params
    parser.add_argument('--amass_train_size', type=int, default=10)
    parser.add_argument('--amass_validation_size', type=int, default=10000)
    parser.add_argument('--amass_test_size', type=int, default=200)
    parser.add_argument('--faust_train_size', type=int, default=10)
    parser.add_argument('--filtering', type=float, default=0, help='amount of filtering to apply on l2 distances')

    #Loss params
    parser.add_argument('--penalty_loss', type=int, default=1, help='penalty applied to points belonging to the mask')



    opt = parser.parse_args()

    opt.manualSeed = 1  # random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)



    # ===================CREATE DATASET================================= #

    dataset_test = AmassProjectionsDataset(type = 'test', num_input_channels=opt.num_input_channels, filtering=opt.filtering,
                                      mask_penalty=opt.penalty_loss, use_same_subject=opt.use_same_subject,
                                      train_size=opt.amass_train_size, validation_size=opt.amass_validation_size)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True,
                                                  num_workers=int(opt.workers), pin_memory=True)

    # ===================CREATE network================================= #
    network = CompletionNet(num_input_channels = opt.num_input_channels)
    network.cuda()  # put network on GPU
    network.apply(weights_init)  # initialization of the weight
    if opt.model_dir != '':
        model_path = os.path.join(os.getcwd(), "log", opt.model_dir, opt.model_file)
        print(model_path)
        network.load_state_dict(torch.load(model_path))
        print(" Previous weight loaded ")

    # ========================================================== #

    dir_name = os.path.join(os.getcwd(), "log", opt.model_dir, opt.save_path)
    print(dir_name)
    os.mkdir(dir_name)
    with torch.no_grad():
        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss_mat, _ = data
            # OH: place on GPU
            part = part.transpose(2, 1).contiguous().cuda().float()
            template = template.transpose(2, 1).contiguous().cuda().float()
            gt = gt.transpose(2, 1).contiguous().cuda().float()

            # Forward pass
            pointsReconstructed, shift_template, shift_part = network(part, template)
            file_name = "subjectIDfull_{}_subjectIDpart_{}_poseIDfull_{}_poseIDpart_{}_projectionID_{}.mat".format(subject_id_full.data[0], subject_id_part.data[0], pose_id_full.data[0], pose_id_part.data[0], mask_id.data[0])
            file_name = os.path.join(os.getcwd(), "log", opt.model_dir, opt.save_path, file_name)
            sio.savemat(file_name, {'pointsReconstructed' : pointsReconstructed.cpu().data.numpy(), 'shift_template' : shift_template.cpu().data.numpy(), 'shift_part' : shift_part.cpu().data.numpy()})
