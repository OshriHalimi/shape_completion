from abc import ABC
from dataset.abstract import ParametricCompletionDataset
from dataset.index import HierarchicalIndexTree
from pickle import load
import numpy as np
from util.mesh.io import read_npz_mask, read_off_verts, read_obj_verts
import scipy.io as sio
import re


# ----------------------------------------------------------------------------------------------------------------------
#                                                     Abstract
# ----------------------------------------------------------------------------------------------------------------------

class FaustProjDataset(ParametricCompletionDataset, ABC):
    def _construct_hit(self):
        hit = {}
        for sub_id in range(10):
            hit[sub_id] = {}
            for pose_id in range(10):
                hit[sub_id][pose_id] = 10
        return HierarchicalIndexTree(hit, in_memory=True)


class AmassProjDataset(ParametricCompletionDataset, ABC):
    def _construct_hit(self):
        hit_appender = re.search(r'Amass(.*)Py', self.__class__.__name__).group(1).lower()
        fp = self._data_dir / f'amass_{hit_appender}_hit.pkl'
        with open(fp, "rb") as f:
            return load(f)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}_projectionID_{hi[2]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / f'subjectID_{hi[0]}_poseID_{hi[1]}.OFF'

    def _proj_path2data(self, fp):
        return read_npz_mask(fp)

    def _full_path2data(self, fp):
        return read_off_verts(fp)


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - Faust
# ----------------------------------------------------------------------------------------------------------------------

class FaustPyProj(FaustProjDataset):
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890, disk_space_bytes=67984856)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'tr_reg_0{hi[0]}{hi[1]}_00{hi[2]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / f'tr_reg_0{hi[0]}{hi[1]}.off'

    def _proj_path2data(self, fp):
        return read_npz_mask(fp)

    def _full_path2data(self, fp):
        return read_off_verts(fp)


class FaustMatProj(FaustProjDataset):
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890, disk_space_bytes=300504051)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'tr_reg_0{hi[0]}{hi[1]}_{hi[2] + 1:03d}.mat'

    def _hi2full_path(self, hi):
        return self._full_dir / f'tr_reg_0{hi[0]}{hi[1]}.mat'

    def _proj_path2data(self, fp):
        return np.squeeze(sio.loadmat(fp)['part_mask'].astype(np.int32) - 1)

    def _full_path2data(self, fp):
        return sio.loadmat(fp)['full_shape']


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - DFaust
# ----------------------------------------------------------------------------------------------------------------------

class DFaustPyProj(ParametricCompletionDataset):
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890,
                         disk_space_bytes=32911290368)

    def _construct_hit(self):
        with open(self._data_dir / 'DFaust_hit.pkl', "rb") as f:
            return load(f)

    def _hi2proj_path(self, hi):
        return self._proj_dir / f'{hi[0]}{hi[1]}{hi[2]:05}_{hi[3]}.npz'

    def _hi2full_path(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:05}.OFF'

    def _proj_path2data(self, fp):
        return read_npz_mask(fp)

    def _full_path2data(self, fp):
        return read_off_verts(fp)


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementations - Amass
# ----------------------------------------------------------------------------------------------------------------------

class AmassTrainPyProj(AmassProjDataset, ABC):
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890,
                         disk_space_bytes=90026754048)


class AmassValdPyProj(AmassProjDataset, ABC):
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890,
                         disk_space_bytes=8288399360)


class AmassTestPyProj(AmassProjDataset, ABC):
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890, disk_space_bytes=691769344)


# ----------------------------------------------------------------------------------------------------------------------
#                                           Implementation - Mixamo
# ----------------------------------------------------------------------------------------------------------------------

class MixamoPyProj(ParametricCompletionDataset):  # Should be: MixamoPyProj_2k_10ang_1fr
    def __init__(self, data_dir_override):
        super().__init__(data_dir_override=data_dir_override, cls='synthetic', n_verts=6890,
                         disk_space_bytes=2.4e+12)

    def _construct_hit(self):
        with open(self._data_dir / 'Mixamo_hit.pkl', "rb") as f:
            return load(f)

    def _hi2proj_path(self, hi):
        for i in range(10):  # Num Angles. Hacky - but works. TODO - Should we rename?
            fp = self._proj_dir / hi[0] / hi[1] / f'{hi[2]:03}_{hi[3]}_angi_{i}.npz'
            if fp.is_file():
                return fp
        else:
            raise AssertionError

    def _hi2full_path(self, hi):
        return self._full_dir / hi[0] / hi[1] / f'{hi[2]:03}.obj'

    def _proj_path2data(self, fp):
        return read_npz_mask(fp)

    def _full_path2data(self, fp):
        return read_obj_verts(fp)


# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------

class FullPartDatasetMenu:
    _implemented = {
        'AmassTestPyProj': AmassTestPyProj,
        'AmassValdPyProj': AmassValdPyProj,
        'AmassTrainPyProj': AmassTrainPyProj,
        'FaustPyProj': FaustPyProj,
        'FaustMatProj': FaustMatProj,
        'DFaustPyProj': DFaustPyProj,
        'MixamoPyProj': MixamoPyProj
    }

    @classmethod
    def which(cls):
        return tuple(cls._implemented.keys())

    @classmethod
    def get(cls, dataset_name, data_dir_override=None):
        return cls._implemented[dataset_name](data_dir_override=data_dir_override)


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------
def test_dataset():
    from dataset.transforms import Center
    ds = FullPartDatasetMenu.get('DFaustPyProj')
    # for ds_name in FullPartDatasetMenu.which():
    #     ds = FullPartDatasetMenu.get(ds_name)
    #     print(ds.num_datapoints_by_method('f2p') , ds._hit_in_memory)
    #     ldr = ds._loader(method='f2p', transforms=None, n_channels=6, ids=None, batch_size=12, device='cpu-single')
    #     for d in ldr:
    #         print(d)
    #         break

    ldr = ds.loaders(s_nums=1000, batch_size=10, device='cpu-single', method='rand_f2p_seq', n_channels=6,
                     s_dynamic=True)
    for d in ldr:
        print(d)
        break
    # ds.data_summary()
    # print(ds.num_projections())
    # print(ds.num_full_shapes())
    # print(ds.num_indexed())
    # for meth in ds.defined_methods():
    #     print(f"{meth} :: {ds.num_datapoints_by_method(meth)} Examples")
    # # ds.validate_dataset()
    # print(ds.num_verts())
    # print(ds.num_faces())
    # print(ds.null_shape(n_channels=12))
    # ds.plot_null_shape()
    # ds.plot_null_shape(strategy='spheres')
    # ds.plot_null_shape(strategy='cloud',with_vnormals=True)
    # def split_loaders(self, s_nums=None, s_shuffle=True, s_transform=(Center(),), split=(1,),
    #                   global_shuffle=False, batch_size=16, device='cuda', method='f2p', n_channels=6)

    # # for meth in ds.defined_methods()
    # for i in range(ds.num_projections()):
    #     for j in range(ds.num_full_shapes()):
    #         if ds._hit.csi2chi(j)[0] == ds._hit.si2hi(i)[0]:  # Same subject
    #             print(ds._hit.csi2chi(j),(i,j),ds._hit.si2hi(i))
    # ldr = ds._loader(method='f2p',transforms=None,n_channels=6,ids=None,batch_size=12,device='cpu-single')
    # for d in ldr:
    #     print(d)
    #     break
    #     print(ldr.num_datapoints())
    #     print(ldr.num_verts())
    #     print(ldr.num_faces())
    #     ldr.plot_null_shape()
    # ldrs = ds.loaders(split=[0.5, 0.3, 0.2], s_shuffle=[False] * 3, s_transform=[Center()] * 3,
    #                   s_nums=[10, 10, 10],
    #                   batch_size=10, device='cpu-single', method='f2p', n_channels=6,
    #                   s_dynamic=[True, False, False])
    # #
    # for ldr in ldrs:
    #     print(ldr.num_indexed())
    #     print(ldr.num_in_iterable())
    #     # d = 0
    #     for c in ldr:
    #         print(c['gt_hi'])
    #         print(c['tp_hi'])


if __name__ == "__main__":
    test_dataset()
