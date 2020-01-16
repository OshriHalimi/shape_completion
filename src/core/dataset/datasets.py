from abc import ABC
from dataset.abstract import SMPLCompletionProjDataset, InCfg
from dataset.index import HierarchicalIndexTree
from pickle import load
import numpy as np
from util.mesh_io import read_npz_mask, read_off_verts
import scipy.io as sio
from dataset.transforms import Center
import re


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
    def _construct_hit(self):
        hit_appender = re.search(r'Amass(.*)Py', self.__class__.__name__).group(1).lower()
        fp = self._data_dir / f'amass_{hit_appender}_hit.pkl'
        with open(fp, "rb") as f:
            return load(f)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}_projectionID_{hi[2]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}.OFF'

    def _proj2data(self, fp):
        return read_npz_mask(fp)

    def _full2data(self, fp):
        return read_off_verts(fp)


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - Faust
# ----------------------------------------------------------------------------------------------------------------------

class FaustPyProj(FaustProjDataset):
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         cls='synthetic', shape=(6890, in_channels), disk_space_bytes=67984856)

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
        super().__init__(data_dir_override=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         cls='synthetic', shape=(6890, in_channels), disk_space_bytes=300504051)

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
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         cls='synthetic', shape=(6890, in_channels), disk_space_bytes=32911290368)

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
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         cls='synthetic', shape=(6890, in_channels), disk_space_bytes=90026754048)


class AmassValdPyProj(AmassProjDataset, ABC):
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         cls='synthetic', shape=(6890, in_channels), disk_space_bytes=8288399360)


class AmassTestPyProj(AmassProjDataset, ABC):
    def __init__(self, in_channels, in_cfg, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, in_channels=in_channels, in_cfg=in_cfg,
                         cls='synthetic', shape=(6890, in_channels), disk_space_bytes=691769344)


# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------

class PointDatasetMenu:
    _implemented = {
        'AmassTestPyProj': AmassTestPyProj,
        'AmassValdPyProj': AmassValdPyProj,
        'AmassTrainPyProj': AmassTrainPyProj,
        'FaustPyProj': FaustPyProj,
        'FaustMatProj': FaustMatProj,
        'DFaustPyProj': DFaustPyProj,
    }

    @staticmethod
    def which():
        return tuple(PointDatasetMenu._implemented.keys())

    @staticmethod
    def get(dataset_name, in_channels=3, in_cfg=InCfg.FULL2PART, data_dir_override=None):
        return PointDatasetMenu._implemented[dataset_name](in_channels=in_channels, in_cfg=in_cfg,
                                                           data_dir_override=data_dir_override)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------
def test_dataset():
    print(PointDatasetMenu.which())
    ds = PointDatasetMenu.get('DFaustPyProj', in_cfg=InCfg.FULL2PART, in_channels=12)
    # ds.data_summary(with_tree=False)
    # ds.validate_dataset()
    # ds.show_sample()
    # print(samp)
    # tl = ds.loader(ids=None,batch_size=2, transforms=[Center()])
    # t1,vl,tsl = ds.split_loaders(split=[0.5,0.4,0.1],s_nums=[100,200,300000],
    # s_shuffle=[True]*3,s_transform=[Center()]*3,global_shuffle=True)
    tl = ds.split_loaders(split=[1], s_nums=[None] , s_shuffle=[True] , s_transform=[Center()],device='cpu-single',batch_size=5)
    #
    # # ids = get_loader_ids(tl)
    from timeit import default_timer as timer
    start = timer()
    for obj in tl:
        obj['gt'].to('cuda')
        obj['tp'].to('cuda')
        end = timer()
        print(f'Load & Transfer Time: {end - start}')
        start = end

    # (trainl, num_train), (validl, num_valid) = ds.testloader()


if __name__ == "__main__":
    test_dataset()
