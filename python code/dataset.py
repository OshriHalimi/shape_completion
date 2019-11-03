from __future__ import print_function
import torch.utils.data as data
from utils import *
import numpy as np
import scipy.io as sio
import json
import os


class IndexExceedDataset(Exception):
    def __init__(self, index, dataset_size):
        self.index = index
        self.dataset_size = dataset_size

    def __str__(self):
        message = "Index exceeds dataset size! index is:{}, size is:{}".format(self.index, self.dataset_size)
        return repr(message)

class SHREC16CutsDavidDataset(data.Dataset):
    def __init__(self):
        #self.path = "D:/oshri.halimi/shape_completion/data/shrec16_evaluation/train_cuts_david/"
        #self.path = "D:/oshri.halimi/shape_completion/data/tosca_plane_cut/david/"
        self.path = "D:/oshri.halimi/shape_completion/data/faust_projections/dataset/"

    def get_shapes(self, index):
        part_id = index + 1
        #name_part = "cuts_david_shape_" + "{}".format(part_id)
        #name_part = "david13_part"
        name_part = "tr_reg_097_001"
        x = sio.loadmat(self.path + name_part + ".mat")
        part = x['partial_shape']  # OH: matrix of vertices

        #name_full = "david"
        #name_full = "david13"
        name_full = "tr_reg_092"
        x = sio.loadmat(self.path + name_full + ".mat")
        template = x['full_shape']  # OH: matrix of vertices

        #part_trans = 0.3*np.random.rand(1,3) - 0.15
        #template_trans = 0.3*np.random.rand(1, 3) - 0.15
        #part[:,:3] = part[:,:3]  + part_trans
        #template[:,:3]  = template[:,:3]  + template_trans

        return part, template, name_part, name_full

    def __getitem__(self, index):
        if index < 1:
            part, template, name_part, name_full = self.get_shapes(index)
        else:
            raise IndexExceedDataset(index, self.__len__())

        return part, template, name_part, name_full, index

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



class AmassProjectionsDataset(data.Dataset):

    def __init__(self, train):
        self.train = train
        if train:
            self.path = os.path.join(os.getcwd(), os.pardir, "data", "train")
            self.dict_counts = json.load(open(os.path.join("support_material", "train_dict.json")))
        else:
            self.path = os.path.join(os.getcwd(), os.pardir, "data", "test")
            self.dict_counts = json.load(open(os.path.join("support_material", "test_dict.json")))
        # self.path = "D:/Shape-Completion/data/faust_projections/dataset/"

    def translate_index(self, index):

        subject_id = np.random.choice(list(map(int, self.dict_counts.keys())))
        pose_id_full, pose_id_part = np.random.choice(self.dict_counts[str(subject_id)], 2, replace=False)
        mask_id = np.random.choice(10)

        return subject_id, pose_id_full, pose_id_part, mask_id

    def get_shapes(self, index):

        subject_id, pose_id_full, pose_id_part, mask_id = self.translate_index(index)

        template = self.read_off(subject_id, pose_id_full)
        gt = self.read_off(subject_id, pose_id_part)
        part = self.read_npz(subject_id, pose_id_part, mask_id)
        part = gt[part]

        return part, template, gt

    def read_npz(self, s_id, p_id, m_id):

        name = os.path.join(self.path, "projection",
                            "subjectID_{}_poseID_{}_projectionID_{}.npz".format(s_id, p_id, m_id))
        mask = np.load(name)
        mask = mask["mask"]

        return mask

    def read_off(self, s_id, p_id):

        name = os.path.join(self.path, "original", "subjectID_{}_poseID_{}.OFF".format(s_id, p_id))

        lines = [l.strip() for l in open(name, "r")]
        words = [int(i) for i in lines[1].split(' ')]
        vn = words[0]
        vertices = np.zeros((vn, 3), dtype='float32')
        for i in range(2, 2 + vn):
            vertices[i - 2] = [float(w) for w in lines[i].split(' ')]

        return vertices

    def __getitem__(self, index):

        part, template, gt = self.get_shapes(index)

        return part, template, gt, index

    def __len__(self):
        if self.train:
            return 100000
        else:
            return 1000


if __name__ == '__main__':
    print('Testing Faust Projections Dataset')

    import visdom
    vis = visdom.Visdom(port=8097, env="main")

    d = AmassProjectionsDataset(train=True)
    i = 11
    part, template, gt, index = d[i]
    vis.scatter(X=template, win="template train sample #{}".format(i),
                opts=dict(title='template train sample #{}'.format(i), markersize=2,),)

    d = AmassProjectionsDataset(train=False)
    i = 11
    part, template, gt, index = d[i]
    vis.scatter(X=template, win="template test sample #{}".format(i),
                opts=dict(title='template test sample #{}'.format(i), markersize=2,),)
