from __future__ import print_function
import torch.utils.data as data
from utils import *
import numpy as np
import scipy.io as sio


class IndexExceedDataset(Exception):
    def __init__(self, index, dataset_size):
        self.index = index
        self.dataset_size = dataset_size

    def __str__(self):
        message = "Index exceeds dataset size! index is:{}, size is:{}".format(self.index, self.dataset_size)
        return repr(message)

class SHREC16CutsDavidDataset(data.Dataset):
    def __init__(self):
        #self.path = "D:/shape_completion/data/shrec16_evaluation/train_cuts_david/"
        self.path = "D:/shape_completion/data/tosca_plane_cut\david/"
        #self.path = "D:/shape_completion/data/faust_projections/dataset/"

    def get_shapes(self, index):
        part_id = index + 1
        #name = "cuts_david_shape_" + "{}".format(part_id)
        name = "david13_part"
        #name = "tr_reg_000_001"
        x = sio.loadmat(self.path + name + ".mat")
        part = x['partial_shape']  # OH: matrix of vertices

        #x = sio.loadmat(self.path + "david.mat")
        x = sio.loadmat(self.path + "david13.mat")
        #x = sio.loadmat(self.path + "tr_reg_000.mat")
        template = x['full_shape']  # OH: matrix of vertices

        return part, template, name

    def __getitem__(self, index):
        if index < 1:
            part, template, name = self.get_shapes(index)
        else:
            raise IndexExceedDataset(index, self.__len__())

        return part, template, name, index

    def __len__(self):
        return 1
class FaustProjectionsDataset(data.Dataset):
    def __init__(self, train):
        self.train = train
        self.path = "D:/Shape-Completion/data/faust_projections/dataset/"

    def translate_index(self, index):
        subject_id = np.floor(index / 1000).astype(int)
        index = index % 1000
        pose_id_full = np.floor(index / 100).astype(int)
        index = index % 100
        pose_id_part = np.floor(index / 10).astype(int)
        index = index % 10
        mask_id = index + 1
        return subject_id, pose_id_full, pose_id_part, mask_id

    def subject_and_pose2shape_ind(self, subject_id, pose_id):
        ind = subject_id * 10 + pose_id
        return ind

    def get_shapes(self, index):
        subject_id, pose_id_full, pose_id_part, mask_id = self.translate_index(index)
        template_id = self.subject_and_pose2shape_ind(subject_id, pose_id_full)
        part_id = self.subject_and_pose2shape_ind(subject_id, pose_id_part)

        x = sio.loadmat(self.path + "tr_reg_" + "{0:0=3d}".format(template_id) + ".mat")
        template = x['full_shape']  # OH: matrix of vertices
        x = sio.loadmat(self.path + "tr_reg_" + "{0:0=3d}".format(part_id) + ".mat")
        gt = x['full_shape']  # OH: matrix of vertices
        x = sio.loadmat(self.path + "tr_reg_" + "{0:0=3d}".format(part_id) + "_" + "{0:0=3d}".format(mask_id) + ".mat")
        part = x['partial_shape']  # OH: matrix of vertices

        return part, template, gt

    def __getitem__(self, index):
        # OH: TODO Consider augmentations such as rotation, translation and downsampling, scale, noise
        if self.train:
            if index < 9000:
                part, template, gt = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())
        else:
            if index < 1000:
                index = index + 9000
                part, template, gt = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())


        #Apply random translation to part and to full template
        part_trans = np.random.rand(1,3) - 0.5
        template_trans = np.random.rand(1, 3) - 0.5
        part[:,:3] = part[:,:3]  + part_trans
        gt[:,:3]  = gt[:,:3] + part_trans
        template[:,:3]  = template[:,:3]  + template_trans

        return part, template, gt, index

    def __len__(self):
        if self.train:
            return 9000
        else:
            return 1000


if __name__ == '__main__':
    print('Testing Faust Projections Dataset')

    import visdom
    vis = visdom.Visdom(port=8888, env="test-rot")

    d = FaustProjectionsDataset(train=True)
    i = 11
    part, template, gt, index = d[i]
    vis.scatter(X=template, win="template train sample #{}".format(i),
                opts=dict(title='template train sample #{}'.format(i), markersize=2,),)

    d = FaustProjectionsDataset(train=False)
    i = 11
    part, template, gt, index = d[i]
    vis.scatter(X=template, win="template test sample #{}".format(i),
                opts=dict(title='template test sample #{}'.format(i), markersize=2,),)
