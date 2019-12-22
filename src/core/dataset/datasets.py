from abc import ABC
from dataset.abstract import SMPLCompletionProjDataset, InCfg
from dataset.index import HierarchicalIndexTree
from pickle import load
import numpy as np
from util.mesh_file import read_npz_mask, read_off_verts
import scipy.io as sio
from dataset.transforms import Center


# ----------------------------------------------------------------------------------------------------------------------
#                                                     Abstract
# ----------------------------------------------------------------------------------------------------------------------

class FaustProjDataset(SMPLCompletionProjDataset, ABC):
    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[sub_id] = {}
            for pose_id in range(10):
                hit[sub_id][pose_id] = 10
        return HierarchicalIndexTree(hit, in_memory=True)


class AmassProjDataset(SMPLCompletionProjDataset, ABC):
    def _hi2proj_path(self, hi):
        return self._proj_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}_projectionID_{hi[2]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / f'"subjectID_{hi[0]}_poseID_{hi[1]}.OFF"'

    def _proj2data(self, fp):
        return read_npz_mask(fp)

    def _full2data(self, fp):
        return read_off_verts(fp)


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - Faust
# ----------------------------------------------------------------------------------------------------------------------

class FaustPyProj(FaustProjDataset):
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         is_synthetic=True, shape=(6890, in_channels), disk_space_bytes=67984856)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'tr_reg_0{hi[0]}{hi[1]}_00{hi[2]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / f'tr_reg_0{hi[0]}{hi[1]}.off'

    def _proj2data(self, fp):
        return read_npz_mask(fp)

    def _full2data(self, fp):
        return read_off_verts(fp)


class FaustMatProj(FaustProjDataset):
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         is_synthetic=True, shape=(6890, in_channels), disk_space_bytes=300504051)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'tr_reg_0{hi[0]}{hi[1]}_{hi[2] + 1:03d}.mat'

    def _hi2full_path(self, hi):
        return self._full_dir / f'tr_reg_0{hi[0]}{hi[1]}.mat'

    def _proj2data(self, fp):
        return np.squeeze(sio.loadmat(fp)['part_mask'].astype(np.int32) - 1)

    def _full2data(self, fp):
        return sio.loadmat(fp)['full_shape']


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

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'{hi[0]}{hi[1]}{hi[2]:05}_{hi[3]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:05}.OFF'

    def _proj2data(self, fp):
        return read_npz_mask(fp)

    def _full2data(self, fp):
        return read_off_verts(fp)


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
        'AmassTestPyProj': AmassTestPyProj,
        'AmassValidPyProj': AmassValidPyProj,
        'AmassTrainPyProj': AmassTrainPyProj,
        'FaustPyProj': FaustPyProj,
        'FaustMatProj': FaustMatProj,
        'DFaustPyProj': DFaustPyProj,
    }

    @staticmethod
    def which():
        return tuple(PointDatasetMenu._implemented.keys())

    @staticmethod
    def get(dataset_name, in_channels=3, in_cfg=InCfg.FULL_FULL_PART, data_dir_overide=None):
        return PointDatasetMenu._implemented[dataset_name](in_channels=in_channels, in_cfg=in_cfg,
                                                           data_dir_overide=data_dir_overide)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------

def test_dataset():
    print(PointDatasetMenu.which())
    ds = PointDatasetMenu.get('FaustMatProj', in_cfg=InCfg.FULL_FULL_PART, in_channels=12)
    ds.data_summary(with_tree=False)
    tl, num_tests = ds.trainloader(valid_rsize=0, transforms=[Center()])
    for obj in tl:
        print(obj)
    # (trainl, num_train), (validl, num_valid) = ds.testloader()


if __name__ == "__main__":
    test_dataset()
