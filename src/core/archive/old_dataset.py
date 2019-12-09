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
from utils import calc_vnrmls, test_normals, normr
from pathlib import Path
from support_material.dfaust_query import generate_dfaust_map





DFAUST_SIDS = ['50002', '50004', '50007', '50009', '50020',
               '50021', '50022', '50025', '50026', '50027']

_, REF_TRI = read_off_full(
    Path(__file__).parents[0] / '..' / 'data' / 'amass' / 'train' / 'original' / 'subjectID_1_poseID_0.OFF')


# -----------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
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


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class DfaustProjectionsDataset(data.Dataset):
    def __init__(self, train, num_input_channels, train_size, test_size, mask_penalty):
        self.train = train
        self.num_input_channels = num_input_channels
        self.path = Path(__file__).parents[0] / '..' / 'data' / 'dfaust'
        self.train_size = train_size
        self.test_size = test_size
        # self.ref_tri = None
        self.mask_penalty = mask_penalty
        self.map = generate_dfaust_map()

    def triangulation(self, use_torch=False):
        # if self.ref_tri is None:
        #     self.ref_tri = REF_TRI

        if use_torch:
            return REF_TRI, torch.from_numpy(REF_TRI).long().cuda()
        else:
            return REF_TRI

    def translate_index(self):
        sid = np.random.choice(10)  # 10 Subjects
        sid_name = DFAUST_SIDS[sid]
        sub_obj = self.map.sub_by_id(sid_name)[0]
        seq_ids = np.random.choice(len(sub_obj.seq_grp), replace=False,
                                   size=(2))  # Don't allow the trivial reconstruction
        frame_gt_name = np.random.choice(sub_obj.frame_cnts[seq_ids[0]])
        frame_temp_name = np.random.choice(sub_obj.frame_cnts[seq_ids[1]])
        seq_gt_name = sub_obj.seq_grp[seq_ids[0]]
        seq_temp_name = sub_obj.seq_grp[seq_ids[1]]
        ang_name = np.random.choice(10)  # 10 Angles

        return sid_name, seq_gt_name, seq_temp_name, frame_gt_name, frame_temp_name, ang_name

    def get_shapes(self):

        sid_name, seq_gt_name, seq_temp_name, frame_gt_name, frame_temp_name, ang_name = self.translate_index()

        gt_fp = self.path / 'unpacked' / sid_name / seq_gt_name / f'{frame_gt_name:05}.OFF'
        template_fp = self.path / 'unpacked' / sid_name / seq_temp_name / f'{frame_temp_name:05}.OFF'
        mask_fp = self.path / 'projections' / f'{sid_name}{seq_gt_name}{frame_gt_name:05}_{ang_name}.npz'

        template = read_off_2(template_fp)
        gt = read_off_2(gt_fp)
        mask = read_npz_2(mask_fp)

        if self.num_input_channels == 6 or self.num_input_channels == 12:
            template_n = calc_vnrmls(template, self.triangulation())
            # test_normals(template, self.triangulation(), template_n)
            gt_n = calc_vnrmls(gt, self.triangulation())
            template = np.concatenate((template, template_n), axis=1)
            gt = np.concatenate((gt, gt_n), axis=1)
            if self.num_input_channels == 12:
                x, y, z = template[:, 0, :], template[:, 1, :], template[:, 2, :]
                template = np.concatenate((template, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
                x, y, z = gt[:, 0, :], gt[:, 1, :], gt[:, 2, :]
                gt = np.concatenate((gt, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)

        mask_loss = np.ones(template.shape[0])
        mask_loss[mask] = self.mask_penalty
        mask_full = np.zeros(template.shape[0], dtype=int)
        mask_full[:len(mask)] = mask
        mask_full[len(mask):] = np.random.choice(mask, template.shape[0] - len(mask), replace=True)
        if len(mask) == 1:
            raise Exception("MASK IS CORRUPTED")

        part = gt[mask_full]

        return template, part, gt, sid_name, sid_name, seq_temp_name, seq_gt_name, ang_name, mask_loss

    def __getitem__(self, index):
        template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss_mat = self.get_shapes()
        return template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss_mat, index

    def __len__(self):
        return self.train_size


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
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
        # self.ref_tri = None

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

    def triangulation(self, use_torch=False):
        # if self.ref_tri is None:
        #     self.ref_tri = REF_TRI

        if use_torch:
            return REF_TRI, torch.from_numpy(REF_TRI).long().cuda()
        else:
            return REF_TRI

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

        if self.num_input_channels == 6 or self.num_input_channels == 12:
            template_n = calc_vnrmls(template, self.triangulation())
            # test_normals(template, self.triangulation(), template_n)
            gt_n = calc_vnrmls(gt, self.triangulation())
            template = np.concatenate((template, template_n), axis=1)
            gt = np.concatenate((gt, gt_n), axis=1)
            if self.num_input_channels == 12:
                x, y, z = template[:, 0, np.newaxis], template[:, 1, np.newaxis], template[:, 2, np.newaxis]
                template = np.concatenate((template, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
                x, y, z = gt[:, 0, np.newaxis], gt[:, 1, np.newaxis], gt[:, 2, np.newaxis]
                gt = np.concatenate((gt, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)

        mask = self.read_npz(subject_id_part, pose_id_part, mask_id)

        mask_loss = np.ones(template.shape[0])
        mask_loss[mask] = self.mask_penalty
        mask_full = np.zeros(template.shape[0], dtype=int)
        mask_full[:len(mask)] = mask
        mask_full[len(mask):] = np.random.choice(mask, template.shape[0] - len(mask), replace=True)
        assert len(mask) > 1, "Mask is Corrupted"
        part = gt[mask_full]

        return template, part, gt, subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id, mask_loss



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


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#
class FaustProjectionsPart2PartDataset(data.Dataset):
    def __init__(self, train, num_input_channels, train_size, test_size):
        self.train = train
        self.num_input_channels = num_input_channels
        self.path = os.path.join(os.getcwd(), os.pardir, "data", "faust_projections", "dataset")
        self.train_size = train_size
        self.test_size = test_size
        # self.ref_tri = None

    def translate_index(self, index):
        subject_id = np.floor(index / 10000).astype(int)
        index = index % 10000
        pose_id_part_1 = np.floor(index / 1000).astype(int)
        index = index % 1000
        pose_id_part_2 = np.floor(index / 100).astype(int)
        index = index % 100
        mask_id_part_1 = np.floor(index / 10).astype(int) + 1
        index = index % 10
        mask_id_part_2 = index + 1
        return subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2

    def triangulation(self, use_torch=False):
        # if self.ref_tri is None:
        #     self.ref_tri = REF_TRI

        if use_torch:
            return REF_TRI, torch.from_numpy(REF_TRI).long().cuda()
        else:
            return REF_TRI

    def subject_and_pose2shape_ind(self, subject_id, pose_id):
        ind = subject_id * 10 + pose_id
        return ind

    def get_shapes(self, index):
        subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2 = self.translate_index(index)
        part_1_id = self.subject_and_pose2shape_ind(subject_id, pose_id_part_1)
        part_2_id = self.subject_and_pose2shape_ind(subject_id, pose_id_part_2)

        x = sio.loadmat(os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_1_id) + ".mat"))
        full_1 = x['full_shape']
        x = sio.loadmat(os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_2_id) + ".mat"))
        full_2 = x['full_shape']
        x = sio.loadmat(
            os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_1_id) + "_" + "{0:0=3d}".format(mask_id_part_1) + ".mat"))

        mask_1 = np.squeeze(x['part_mask'] - 1)  # -1 Matlab indices start with 1 and not 0

        x = sio.loadmat(
            os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_2_id) + "_" + "{0:0=3d}".format(mask_id_part_2) + ".mat"))

        mask_2 = np.squeeze(x['part_mask'] - 1)  # -1 Matlab indices start with 1 and not 0

        if self.num_input_channels == 12:
            x, y, z = full_1[:,0,np.newaxis], full_1[:,1,np.newaxis], full_1[:,2,np.newaxis]
            full_1 = np.concatenate((full_1, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
            x, y, z = full_2[:,0,np.newaxis], full_2[:,1,np.newaxis], full_2[:,2,np.newaxis]
            full_2 = np.concatenate((gt, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)

        mask_full_1 = np.zeros(full_1.shape[0], dtype=int)
        mask_full_1[:len(mask_1)] = mask_1
        mask_full_1[len(mask_1):] = np.random.choice(mask_1, full_1.shape[0] - len(mask_1), replace=True)
        part_1 = full_1[mask_full_1]

        mask_full_2 = np.zeros(full_2.shape[0], dtype=int)
        mask_full_2[:len(mask_2)] = mask_2
        mask_full_2[len(mask_2):] = np.random.choice(mask_2, full_2.shape[0] - len(mask_2), replace=True)
        part_2 = full_2[mask_full_2]

        gt = full_2[mask_full_1]

        assert len(mask_1) > 1, "Mask of Part 1 is Corrupted"
        assert len(mask_2) > 1, "Mask of Part 2 is Corrupted"

        return part_1, part_2, gt, subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2

    def __getitem__(self, index):
        # OH: TODO Consider augmentations such as rotation, translation and downsampling, scale, noise
        if self.train:
            if index < self.train_size:
                part_1, part_2, gt, subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2 = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())
        else:
            if index < self.test_size:
                index = index + self.train_size
                part_1, part_2, gt, subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2 = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())

        # Apply random translation to part and to full template
        # part_trans = np.random.rand(1,3) - 0.5
        # template_trans = np.random.rand(1, 3) - 0.5
        # part[:,:3] = part[:,:3]  + part_trans
        # gt[:,:3]  = gt[:,:3] + part_trans
        # template[:,:3]  = template[:,:3]  + template_trans

        if self.num_input_channels == 3:
            part_1 = part_1[:, :self.num_input_channels]
            part_2 = part_2[:, :self.num_input_channels]
            gt = gt[:, :self.num_input_channels]

        return part_1, part_2, gt, subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2, index

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size


class FaustProjectionsDataset(data.Dataset):
    def __init__(self, train, num_input_channels, train_size, test_size, mask_penalty):
        self.train = train
        self.num_input_channels = num_input_channels
        self.path = os.path.join(os.getcwd(), os.pardir, "data", "faust_projections", "dataset")
        self.train_size = train_size
        self.test_size = test_size
        # self.ref_tri = None
        self.mask_penalty = mask_penalty

    def translate_index(self, index):
        subject_id = np.floor(index / 1000).astype(int)
        index = index % 1000
        pose_id_full = np.floor(index / 100).astype(int)
        index = index % 100
        pose_id_part = np.floor(index / 10).astype(int)
        index = index % 10
        mask_id = index + 1
        return subject_id, pose_id_full, pose_id_part, mask_id

    def triangulation(self, use_torch=False):
        # if self.ref_tri is None:
        #     self.ref_tri = REF_TRI

        if use_torch:
            return REF_TRI, torch.from_numpy(REF_TRI).long().cuda()
        else:
            return REF_TRI

    def subject_and_pose2shape_ind(self, subject_id, pose_id):
        ind = subject_id * 10 + pose_id
        return ind

    def get_shapes(self, index):
        subject_id, pose_id_full, pose_id_part, mask_id = self.translate_index(index)
        template_id = self.subject_and_pose2shape_ind(subject_id, pose_id_full)
        part_id = self.subject_and_pose2shape_ind(subject_id, pose_id_part)

        x = sio.loadmat(os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(template_id) + ".mat"))
        template = x['full_shape']
        x = sio.loadmat(os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_id) + ".mat"))
        gt = x['full_shape']
        x = sio.loadmat(
            os.path.join(self.path, "tr_reg_" + "{0:0=3d}".format(part_id) + "_" + "{0:0=3d}".format(mask_id) + ".mat"))

        mask = np.squeeze(x['part_mask'] - 1)  # -1 - Oshri forgot that Matlab indices start with 1 and not 0
        mask_loss = np.ones(template.shape[0])
        mask_loss[mask] = self.mask_penalty

        if self.num_input_channels == 12:
            x, y, z = template[:,0,np.newaxis], template[:,1,np.newaxis], template[:,2,np.newaxis]
            template = np.concatenate((template, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
            x, y, z = gt[:,0,np.newaxis], gt[:,1,np.newaxis], gt[:,2,np.newaxis]
            gt = np.concatenate((gt, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
            mask_full = np.zeros(template.shape[0], dtype=int)
            mask_full[:len(mask)] = mask
            mask_full[len(mask):] = np.random.choice(mask, template.shape[0] - len(mask), replace=True)
            part = gt[mask_full]
        else:
            part = x['partial_shape']  # OH: matrix of vertices

        assert len(mask) > 1, "Mask is Corrupted"

        return part, template, gt, mask_loss

    def __getitem__(self, index):
        # OH: TODO Consider augmentations such as rotation, translation and downsampling, scale, noise
        if self.train:
            if index < self.train_size:
                part, template, gt, mask_loss = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())
        else:
            if index < self.test_size:
                index = index + self.train_size
                part, template, gt, mask_loss = self.get_shapes(index)
            else:
                raise IndexExceedDataset(index, self.__len__())

        # Apply random translation to part and to full template
        # part_trans = np.random.rand(1,3) - 0.5
        # template_trans = np.random.rand(1, 3) - 0.5
        # part[:,:3] = part[:,:3]  + part_trans
        # gt[:,:3]  = gt[:,:3] + part_trans
        # template[:,:3]  = template[:,:3]  + template_trans
        if self.num_input_channels == 3:
            template = template[:, :self.num_input_channels]
            part = part[:, :self.num_input_channels]
            gt = gt[:, :self.num_input_channels]

        return part, template, gt, index, mask_loss

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size


if __name__ == '__main__':
    print('AMASS Projections Dataset')

    import visdom

    vis = visdom.Visdom(port=8888, env="test-amass-dataset")
    n_input_ch = 6

    d = FaustProjectionsPart2PartDataset(train=True, num_input_channels=6, train_size=100000, test_size=100000)
    for i in range(13903,13913):
        part_1, part_2, gt, subject_id, pose_id_part_1, pose_id_part_2, mask_id_part_1, mask_id_part_2, index = d[i]
        vis.scatter(X=part_1[:, :3], win=f"part_1 train sample #{i}",
                    opts=dict(title=f"part_1 train sample #{i}", markersize=2))
        vis.scatter(X=part_2[:, :3], win=f"part_2 train sample #{i}",
                    opts=dict(title=f"part_2 train sample #{i}", markersize=2))
        vis.scatter(X=gt[:, :3], win=f"ground truth train sample #{i}",
                    opts=dict(title=f"ground truth train sample #{i}", markersize=2))
        print("SID.{}".format(subject_id))
        print("PID1.{}".format(pose_id_part_1))
        print("PID2.{}".format(pose_id_part_2))
        print("MID1.{}".format(mask_id_part_1))
        print("MID2.{}".format(mask_id_part_2))


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


# ----------------------------------------------------------------------------------------------------------------------#
#
# ----------------------------------------------------------------------------------------------------------------------#


def read_off_2(fp):
    lines = [l.strip() for l in open(fp, "r")]
    words = [int(i) for i in lines[1].split(' ')]
    vn = words[0]
    vertices = np.zeros((vn, 3), dtype='float32')
    for i in range(2, 2 + vn):
        vertices[i - 2] = [float(w) for w in lines[i].split(' ')]

    return vertices


def read_npz_2(fp):
    return np.load(fp)['mask']
