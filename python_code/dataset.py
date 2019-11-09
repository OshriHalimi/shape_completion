from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import scipy.io as sio
import json
import os
import time
import scipy
from numpy.matlib import repmat
from utils import compute_vertex_normals, test_normals,normr


class IndexExceedDataset(Exception):
    def __init__(self, index, dataset_size):
        self.index = index
        self.dataset_size = dataset_size

    def __str__(self):
        message = "Index exceeds dataset size! index is:{}, size is:{}".format(self.index, self.dataset_size)
        return repr(message)


class SHREC16CutsDavidDataset(data.Dataset):
    def __init__(self):
        # self.path = "D:/oshri.halimi/shape_completion/data/shrec16_evaluation/train_cuts_david/"
        # self.path = "D:/oshri.halimi/shape_completion/data/tosca_plane_cut/david/"
        self.path = "D:/oshri.halimi/shape_completion/data/faust_projections/dataset/"

    def get_shapes(self, index):
        part_id = index + 1
        # name_part = "cuts_david_shape_" + "{}".format(part_id)
        # name_part = "david13_part"
        name_part = "tr_reg_097_001"
        x = sio.loadmat(self.path + name_part + ".mat")
        part = x['partial_shape']  # OH: matrix of vertices

        # name_full = "david"
        # name_full = "david13"
        name_full = "tr_reg_092"
        x = sio.loadmat(self.path + name_full + ".mat")
        template = x['full_shape']  # OH: matrix of vertices

        # part_trans = 0.3*np.random.rand(1,3) - 0.15
        # template_trans = 0.3*np.random.rand(1, 3) - 0.15
        # part[:,:3] = part[:,:3]  + part_trans
        # template[:,:3]  = template[:,:3]  + template_trans

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
    def __init__(self, train, num_input_channels, train_size):
        self.train = train
        self.num_input_channels = num_input_channels
        self.path = os.path.join(os.getcwd(), os.pardir, "data", "faust_projections", "dataset")
        self.train_size = train_size  # was 9000 when we train on FaustProjectionsDataset, but we set it to 10000 (full size: train and test) when we use it for evaluation
        self.test_size = 1000

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

        x = sio.loadmat(os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(template_id) + ".mat"))
        template = x['full_shape']  # OH: matrix of vertices
        x = sio.loadmat(os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_id) + ".mat"))
        gt = x['full_shape']  # OH: matrix of vertices
        x = sio.loadmat(
            os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_id) + "_" + "{0:0=3d}".format(mask_id) + ".mat"))
        part = x['partial_shape']  # OH: matrix of vertices

        return part, template, gt

    def __getitem__(self, index):
        # OH: TODO Consider augmentations such as rotation, translation and downsampling, scale, noise
        if self.train:
            if index < self.train_size:
                part, template, gt = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())
        else:
            if index < self.test_size:
                index = index + self.train_size
                part, template, gt = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())

        # Apply random translation to part and to full template
        # part_trans = np.random.rand(1,3) - 0.5
        # template_trans = np.random.rand(1, 3) - 0.5
        # part[:,:3] = part[:,:3]  + part_trans
        # gt[:,:3]  = gt[:,:3] + part_trans
        # template[:,:3]  = template[:,:3]  + template_trans

        template = template[:, :self.num_input_channels]
        part = part[:, :self.num_input_channels]
        gt = gt[:, :self.num_input_channels]

        return part, template, gt, index

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size


class AmassProjectionsDataset(data.Dataset):
    def __init__(self, split, num_input_channels, filtering, mask_penalty,
                 use_same_subject=True, train_size=100000, validation_size=10000, test_size=300):
        self.split = split
        self.num_input_channels = num_input_channels
        self.use_same_subject = use_same_subject
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.filtering = filtering
        self.mask_penalty = mask_penalty
        self.ref_tri = None

        if self.split == 'train':
            self.path = os.path.join(os.getcwd(), os.pardir, "data", "amass", "train")
            print("Train set path:")
            print(self.path)
            self.dict_counts = json.load(open(os.path.join("support_material", "train_dict.json")))
        if self.split == 'validation':
            self.path = os.path.join(os.getcwd(), os.pardir, "data", "amass", "vald")
            print("Validation set path:")
            print(self.path)
            self.dict_counts = json.load(open(os.path.join("support_material", "vald_dict.json")))
        if self.split == 'test':
            self.path = os.path.join(os.getcwd(), os.pardir, "data", "amass", "test")
            print("Test set path:")
            print(self.path)
            self.dict_counts = json.load(open(os.path.join("support_material", "test_dict.json")))

    def triangulation(self):
        if self.ref_tri is None:
            ref_fp = os.path.join(self.path, "original", "subjectID_1_poseID_0.OFF")
            _, self.ref_tri = self.read_off_full(ref_fp)
        # if use_torch:
        #     return torch.LongTensor(self.ref_tri).cuda()
        # else:
        return self.ref_tri


    def translate_index(self):

        subject_id_full = np.random.choice(list(map(int, self.dict_counts.keys())))
        while subject_id_full == 288:  # this needs to be fixed/removed (has only one pose???)
            subject_id_full = np.random.choice(list(map(int, self.dict_counts.keys())))

        if self.use_same_subject == True:
            subject_id_part = subject_id_full
        else:
            subject_id_part = np.random.choice(list(map(int, self.dict_counts.keys())))
            while subject_id_part == 288:  # this needs to be fixed/removed (has only one pose???)
                subject_id_part = np.random.choice(list(map(int, self.dict_counts.keys())))

        pose_id_full = np.random.choice(self.dict_counts[str(subject_id_full)])
        pose_id_part = np.random.choice(self.dict_counts[str(subject_id_part)])
        mask_id = np.random.choice(10)

        return subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id

    def get_shapes(self):

        subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id = self.translate_index()
        template = self.read_off(subject_id_full, pose_id_full)
        gt = self.read_off(subject_id_part, pose_id_part)
        euc_dist = np.mean((template - gt) ** 2)

        if self.filtering > 0:
            if self.use_same_subject == True:
                while np.random.rand() > (euc_dist / self.filtering):
                    subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id = self.translate_index()
                    template = self.read_off(subject_id_full, pose_id_full)
                    gt = self.read_off(subject_id_part, pose_id_part)
                    euc_dist = np.mean((template - gt) ** 2)

        if self.num_input_channels == 6:
            template_n = compute_vertex_normals(template, self.triangulation())
            # test_normals(template, self.triangulation(), template_n)
            gt_n = compute_vertex_normals(gt, self.triangulation())
            template = np.concatenate((template, template_n), axis=1)
            gt = np.concatenate((gt, gt_n), axis=1)

        mask = self.read_npz(subject_id_part, pose_id_part, mask_id)
        # mask_loss_mat = np.ones((template.shape[0], template.shape[1]), dtype=int)
        mask_loss = np.ones(template.shape[0])
        mask_loss[mask] = self.mask_penalty
        # mask_loss_mat[:, 0] = mask_loss
        # mask_loss_mat[:, 1] = mask_loss
        # mask_loss_mat[:, 2] = mask_loss
        mask_full = np.zeros(template.shape[0], dtype=int)
        mask_full[:len(mask)] = mask
        mask_full[len(mask):] = np.random.choice(mask, template.shape[0] - len(mask), replace=True)
        part = gt[mask_full]

        if len(mask) == 1:
            raise Exception("MASK IS CORRUPTED")

        return template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss

    def read_npz(self, s_id, p_id, m_id):

        name = os.path.join(self.path, "projection",
                            "subjectID_{}_poseID_{}_projectionID_{}.npz".format(s_id, p_id, m_id))
        mask = np.load(name)
        mask = mask["mask"]

        return mask

    def read_off_full(self, off_file):
        vertexBuffer = []
        indexBuffer = []
        with open(off_file, "r") as modelfile:
            first = modelfile.readline().strip()
            if first != "OFF":
                raise (Exception("not a valid OFF file ({})".format(first)))

            parameters = modelfile.readline().strip().split()

            if len(parameters) < 2:
                raise (Exception("OFF file has invalid number of parameters"))

            for i in range(int(parameters[0])):
                coordinates = modelfile.readline().split()
                vertexBuffer.append([float(coordinates[0]), float(coordinates[1]), float(coordinates[2])])

            for i in range(int(parameters[1])):
                indices = modelfile.readline().split()
                indexBuffer.append([int(indices[1]), int(indices[2]), int(indices[3])])

        return np.array(vertexBuffer), np.array(indexBuffer)

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

        template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss_mat = self.get_shapes()

        return template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss_mat, index

    def __len__(self):
        if self.split == 'train':
            return self.train_size
        if self.split == 'validation':
            return self.validation_size
        if self.split == 'test':
            return self.test_size


if __name__ == '__main__':
    print('AMASS Projections Dataset')

    import visdom

    vis = visdom.Visdom(port=8888, env="test-amass-dataset")
    n_input_ch = 6

    d = FaustProjectionsDataset(train=True, num_input_channels=6, train_size=100)
    for i in range(10):
        part, template, gt, index = d[i]

    d = AmassProjectionsDataset(train=True, num_input_channels=n_input_ch, use_same_subject=True)
    for i in range(10):
        part, template, gt, index = d[i]
        vis.scatter(X=template[:, :3], win=f"template train sample #{i}",
                    opts=dict(title=f'template train sample #{i}', markersize=2))

        # if d.num_input_channels == 6:
        #     vis.quiver(template[:, :3], template[:, :3] + template[:, 3:6], win=f"template train sample #{i}",
        #                opts=dict(title=f'template train sample #{i}', markersize=2),)

    d = AmassProjectionsDataset(train=False, num_input_channels=n_input_ch, use_same_subject=True)
    for i in range(10):
        part, template, gt, index = d[i]
        vis.scatter(X=template[:, :3], win=f"template train sample #{i}",
                    opts=dict(title=f'template train sample #{i}', markersize=2))

        # if d.num_input_channels == 6:
        #     vis.quiver(template[:, :3], template[:, :3] + template[:, 3:6], win=f"template train sample #{i}",
        #                opts=dict(title=f'template train sample #{i}', markersize=2))
