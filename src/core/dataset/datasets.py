from abc import ABC
from dataset.abstract import SMPLCompletionProjDataset, HierarchicalIndexTree
from collections import defaultdict
from pickle import dump, load
import numpy as np
from util.mesh_file import read_npz_mask, read_off_verts
import scipy.io as sio
from dataset.transforms import CompletionPair, Center


# ----------------------------------------------------------------------------------------------------------------------
#                                                     TODO
# ----------------------------------------------------------------------------------------------------------------------
# euc_dist = np.mean((template - gt) ** 2)
#
# if self.filtering > 0:
#     if self.use_same_subject == True:
#         while np.random.rand() > (euc_dist / self.filtering):
#             subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id = self.translate_index()
#             template = self.read_off(subject_id_full, pose_id_full)
#             gt = self.read_off(subject_id_part, pose_id_part)
#             euc_dist = np.mean((template - gt) ** 2)
# ----------------------------------------------------------------------------------------------------------------------
#                                                     Abstract
# ----------------------------------------------------------------------------------------------------------------------

class FaustProjDataset(SMPLCompletionProjDataset, ABC):
    def _construct_hit(self):
        hit = defaultdict(dict)
        for sub_id in range(10):
            for pose_id in range(10):
                hit[sub_id][pose_id] = 10
        return HierarchicalIndexTree(hit, in_memory=True)


class AmassProjDataset(SMPLCompletionProjDataset, ABC):
    def _path_load(self, fps):
        pass
        # template = self.read_off(subject_id_full, pose_id_full)
        # gt = self.read_off(subject_id_part, pose_id_part)
        # mask = self.read_npz(subject_id_part, pose_id_part, mask_id)

    def _hierarchical_index_to_path(self, hi):  # TODO - Make this a single file load
        pass

    # subject_id_full = np.random.choice(list(map(int, self.dict_counts.keys())))
    #  while subject_id_full == 288:  # this needs to be fixed/removed (has only one pose???)
    #      subject_id_full = np.random.choice(list(map(int, self.dict_counts.keys())))
    #
    #  if self.use_same_subject == True:
    #      subject_id_part = subject_id_full
    #  else:
    #      subject_id_part = np.random.choice(list(map(int, self.dict_counts.keys())))
    #      while subject_id_part == 288:  # this needs to be fixed/removed (has only one pose???)
    #          subject_id_part = np.random.choice(list(map(int, self.dict_counts.keys())))
    #
    #  pose_id_full = np.random.choice(self.dict_counts[str(subject_id_full)])
    #  pose_id_part = np.random.choice(self.dict_counts[str(subject_id_part)])
    #  mask_id = np.random.choice(10)
    #
    #  return subject_id_full, subject_id_part, pose_id_full, pose_id_part, mask_id


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - Faust
# ----------------------------------------------------------------------------------------------------------------------

class FaustPyProj(FaustProjDataset):
    def __init__(self, in_channels, validate, data_dir_override):
        super().__init__(shape=((6890, in_channels), (6890, in_channels)), disk_space_bytes=67984856, is_synthetic=True,
                         num_disk_accesses=2, existing_in_channels=3, in_channels=in_channels,
                         data_dir=data_dir_override, validate=validate)

    def _hierarchical_index_to_path(self, hi):  # TODO - Make this a single file load
        return [self._data_dir / 'projections' / f'tr_reg_0{hi[0]}{hi[1]}_00{hi[2]}.npz',
                self._data_dir / 'full' / f'tr_reg_0{hi[0]}{hi[1]}.off', hi]

    def _path_load(self, fps):
        mask_vi = read_npz_mask(fps[0])  # Starts at 0
        gt_v = read_off_verts(fps[1])
        return CompletionPair(gt_v, mask_vi, fps[2], self._f)


class FaustMatProj(FaustProjDataset):
    def __init__(self, in_channels, validate, data_dir_override):
        super().__init__(shape=((6890, in_channels), (6890, in_channels)), disk_space_bytes=300504051,
                         is_synthetic=True, num_disk_accesses=2,
                         existing_in_channels=6, in_channels=in_channels, data_dir=data_dir_override, validate=validate)

    def _hierarchical_index_to_path(self, hi):  # TODO - Make this a single file load
        return [self._data_dir / 'projections' / f'tr_reg_0{hi[0]}{hi[1]}_{hi[2] + 1:03d}.mat',
                self._data_dir / 'full' / f'tr_reg_0{hi[0]}{hi[1]}.mat', hi]

    def _path_load(self, fps):
        # First container also holds 'partial_shape'
        mask_vi = np.squeeze(sio.loadmat(fps[0])['part_mask'].astype(np.int32) - 1)
        gt_v = sio.loadmat(fps[1])['full_shape']
        return CompletionPair(gt_v, mask_vi, fps[2])


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - DFaust
# ----------------------------------------------------------------------------------------------------------------------

class DFaustPyProj(SMPLCompletionProjDataset):
    def __init__(self, in_channels, validate, data_dir_override):
        super().__init__(shape=((6890, in_channels), (6890, in_channels)), disk_space_bytes=-1,
                         is_synthetic=True, num_disk_accesses=-1,
                         existing_in_channels=-1, in_channels=in_channels, data_dir=data_dir_override,
                         validate=validate)

    def _construct_hit(self):
        with open(self._data_dir / 'DFaust_hit.pkl', "rb") as f:
            return load(f)

    def _hierarchical_index_to_path(self, hi):  # TODO - Make this a single file load
        pass
        # gt_fp = self.path / 'unpacked' / sid_name / seq_gt_name / f'{frame_gt_name:05}.OFF'
        # template_fp = self.path / 'unpacked' / sid_name / seq_temp_name / f'{frame_temp_name:05}.OFF'
        # mask_fp = self.path / 'projections' / f'{sid_name}{seq_gt_name}{frame_gt_name:05}_{ang_name}.npz'

    def _path_load(self, fps):
        pass
        # template = read_off_2(template_fp)
        # gt = read_off_2(gt_fp)
        # mask = read_npz_2(mask_fp)

    # sid = np.random.choice(10)  # 10 Subjects
    # sid_name = DFAUST_SIDS[sid]
    # sub_obj = self.map.sub_by_id(sid_name)[0]
    # seq_ids = np.random.choice(len(sub_obj.seq_grp), replace=False,
    #                            size=(2))  # Don't allow the trivial reconstruction
    # frame_gt_name = np.random.choice(sub_obj.frame_cnts[seq_ids[0]])
    # frame_temp_name = np.random.choice(sub_obj.frame_cnts[seq_ids[1]])
    # seq_gt_name = sub_obj.seq_grp[seq_ids[0]]
    # seq_temp_name = sub_obj.seq_grp[seq_ids[1]]
    # ang_name = np.random.choice(10)  # 10 Angles


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - Amass
# ----------------------------------------------------------------------------------------------------------------------

class AmassTrainPyProj(AmassProjDataset, ABC):
    def _construct_hit(self):
        with open(self._data_dir / 'DFaust_hit.pkl', "rb") as f:
            return load(f)


class AmassValidPyProj(AmassProjDataset, ABC):
    def _construct_hit(self):
        with open(self._data_dir / 'DFaust_hit.pkl', "rb") as f:
            return load(f)


class AmassTestPyProj(AmassProjDataset, ABC):
    def _construct_hit(self):
        with open(self._data_dir / 'DFaust_hit.pkl', "rb") as f:
            return load(f)


# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------

class PointDatasetMenu:
    _implemented = {
        # 'AmassTrain': AmassTrain,
        # 'AmassValid': AmassValid,
        # 'AmassTest': FaustPyProj,
        'FaustPyProj': FaustPyProj,
        'FaustMatProj': FaustMatProj,
        'DFaustPyProj': DFaustPyProj,
    }

    @staticmethod
    def which():
        return tuple(PointDatasetMenu._implemented.keys())

    @staticmethod
    def get(dataset_name, in_channels=3, validate=False, data_dir_overide=None):
        return PointDatasetMenu._implemented[dataset_name](in_channels, validate, data_dir_overide)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------

def test_dataset():
    print(PointDatasetMenu.which())
    ds = PointDatasetMenu.get('FaustMatProj', validate=False, in_channels=12)
    ds.data_summary(with_tree=False)
    tl, num_tests = ds.trainloader(valid_rsize=0, transforms=[Center()])
    for obj in tl:
        print(obj)
    (trainl, num_train), (validl, num_valid) = ds.testloader()


if __name__ == "__main__":
    test_dataset()
