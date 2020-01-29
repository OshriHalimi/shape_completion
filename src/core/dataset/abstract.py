from torch.utils.data import Dataset
from pathlib import Path
from dataset.collate import completion_collate
from collections import Sequence
from abc import ABC  # , abstractmethod
from torch.utils.data.sampler import SubsetRandomSampler
from util.torch_data_ext import determine_worker_num, ReconstructableDataLoader
from util.gen import warn, banner, convert_bytes, time_me
from util.container import split_frac, enum_eq
from util.mesh_compute import trunc_to_vertex_subset
from pickle import load
from copy import deepcopy
from dataset.transforms import *
from enum import Enum, auto
import sys
from tqdm import tqdm
from types import MethodType
import time
from timeit import default_timer as timer


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
# TODO - Change this to a standard class, allowing for more configuration
class InCfg(Enum):
    FULL2PART = auto()
    PART2PART = auto()


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class PointDataset(ABC):
    def __init__(self, data_dir_override, cls, shape, in_channels, disk_space_bytes):
        # Check Data Directory:
        if data_dir_override is None:
            from cfg import PRIMARY_DATA_DIR
            self._data_dir = PRIMARY_DATA_DIR / cls / self.name()
        else:
            self._data_dir = Path(data_dir_override)
        assert self._data_dir.is_dir(), f"Data directory of {self.name()} is invalid: \nCould not find {self._data_dir}"
        self._data_dir = self._data_dir.resolve()

        # Dirty and hacky - Shouldn't be here, but it is the simplest.
        self._in_cfg = None
        # Basic Dataset Info
        self._hit = self._construct_hit()

        # Insert Info:
        self._shape = tuple(shape)
        from cfg import SUPPORTED_IN_CHANNELS
        assert in_channels in SUPPORTED_IN_CHANNELS
        self._in_channels = in_channels
        self._disk_space_bytes = disk_space_bytes

        self._f = None

    def data_summary(self, with_tree=False):
        banner('Dataset Summary')
        print(f'* Dataset Name: {self.name()}')
        print(f'* Point Cloud Number: {self.num_pnt_clouds()}')
        print(f'* Singleton Input Data Shape: {self._shape}')
        print(f'* Number of Input Channels Requested: {self._in_channels}')
        print(f'* Estimated Hard-disk space required: ~{convert_bytes(self._disk_space_bytes)}')
        print(f'* Direct Filepath: {self._data_dir}')
        if with_tree:
            banner('Dataset Index Tree')
            self.report_index_tree()
        banner()

    def name(self):
        return self.__class__.__name__

    def num_pnt_clouds(self):
        return self._hit.num_objects()

    def shape(self):
        return self._shape

    def disk_space(self):
        return self._disk_space_bytes

    def report_index_tree(self):
        print(self._hit)

    def faces(self):
        assert self._f is not None, "Faces property is empty"
        return deepcopy(self._f)

    def sample(self, num_samples=10, transforms=None):
        if num_samples > self.num_pnt_clouds():
            warn(f"Requested {num_samples} samples when dataset only holds {self.num_pnt_clouds()}. "
                 f"Returning the latter")
        ldr = self._loader(ids=None, transforms=transforms, batch_size=num_samples, device='cpu-single')
        return next(iter(ldr))
        # return [attempt_squeeze(next(ldr_it)) for _ in range(num_samples)]

    def _loader(self, ids=None, transforms=None, batch_size=16, device='cuda'):
        # TODO - Add distributed support here. What does num_workers need to be?
        if ids is None:
            ids = range(self.num_pnt_clouds())
        assert len(ids) > 0, "Found loader with no data samples inside"

        device = str(device).split(':')[0]  # Compatible for both strings & pytorch devs
        assert device in ['cuda', 'cpu', 'cpu-single']
        pin_memory = (device == 'cuda')
        if device == 'cpu-single':
            n_workers = 0
        else:
            n_workers = determine_worker_num(len(ids), batch_size)

        # if self.use_ddp:
        #     train_sampler = DistributedSampler(dataset)

        train_loader = ReconstructableDataLoader(self._set_to_torch_set(transforms, len(ids)), batch_size=batch_size,
                                                 sampler=SubsetRandomSampler(ids), num_workers=n_workers,
                                                 pin_memory=pin_memory, collate_fn=completion_collate)
        return train_loader

    def split_loaders(self, s_nums=None, s_shuffle=True, s_transform=(Center(),), split=(1,),
                      global_shuffle=False, batch_size=16, device='cuda'):
        """
        # s for split
        :param split: A list of fracs summing to 1: e.g.: [0.9,0.1] or [0.8,0.15,0.05]. Don't specify anything for a
        single loader
        :param s_nums: A list of integers: e.g. [1000,50] or [1000,5000,None] - The number of objects to take from each
        range split. If None, it will take the maximal number possible
        :param s_shuffle: A list of booleans: If s_shuffle[i]==True, the ith split will be shuffled before truncations
        to s_nums[i] objects
        :param s_transform: A list - s_transforms[i] is the transforms for the ith split
        :param global_shuffle: If True, shuffles the entire set before split
        :param batch_size: Integer > 0
        :param device: 'cuda' or 'cpu' or 'cpu-single' or pytorch device
        :return: A list of (loaders,num_samples)
        """
        # Handle inpput arguments:
        if not isinstance(s_shuffle, Sequence):
            s_shuffle = [s_shuffle]
        if not isinstance(s_nums, Sequence):
            s_nums = [s_nums]
        if s_transform is None:
            s_transform = []
            # Transforms must be a list, all others are non-Sequence
        assert sum(split) == 1, "Split fracs must sum to 1"

        # Logic:
        ids = list(range(self.num_pnt_clouds()))
        if global_shuffle:
            np.random.shuffle(ids)  # Mixes up the whole set

        n_parts = len(split)
        ids = split_frac(ids, split)
        loaders = []
        for i in range(n_parts):
            set_ids, req_set_size, do_shuffle, transforms = ids[i], s_nums[i], s_shuffle[i], s_transform[i]
            if req_set_size is None:
                req_set_size = len(set_ids)
            eff_set_size = min(len(set_ids), req_set_size)
            if eff_set_size != req_set_size:
                warn(f'At Loader {i + 1}/{n_parts}: Requested {req_set_size} objects while set has only {eff_set_size}.'
                     f' Reverting to latter')
            if do_shuffle:
                np.random.shuffle(set_ids)  # Truncated sets may now hold different ids
            set_ids = set_ids[:eff_set_size]  # Truncate
            recon_stats = {
                'dataset_name': self.name(),
                'batch_size': batch_size,
                'split': split,
                'id_in_split': i,
                'set_size': eff_set_size,
                'transforms': str(transforms),
                'global_shuffle': global_shuffle,
                'partition_shuffle': do_shuffle}
            if self._in_cfg is not None:
                recon_stats['in_cfg'] = self._in_cfg.name
            ldr = self._loader(ids=set_ids, transforms=transforms, batch_size=batch_size, device=device)
            ldr.init_recon_table(recon_stats)
            loaders.append(ldr)

        if n_parts == 1:
            loaders = loaders[0]
        return loaders

    @time_me
    def validate_dataset(self):
        banner(f'Validation of dataset {self.name()} :: {self.num_pnt_clouds()} pnt clouds')
        # time.sleep(.01)  # For the STD-ERR lag
        for si in tqdm(range(self.num_pnt_clouds()), file=sys.stdout, dynamic_ncols=True):
            hi = self._hit.si2hi(si)
            fps = self._hierarchical_index_to_path(hi)
            if not isinstance(fps, list):
                fps = [fps]
            # TODO:
            # (1) Maybe add a count of possible missing files? Not really needed, seeing working on a partial dataset
            # requires updating the hit
            # (2) Note that it is not really needed to iterate over all the fps - only the projections + full set is
            # enough - Better to change the call to something different maybe?
            found_valid_fp = False
            for fp in fps:
                if isinstance(fp, Path):
                    found_valid_fp = True
                    assert fp.is_file(), f"Missing file {fp.resolve()} in dataset {self.name()}"
            assert found_valid_fp, "Filepaths are not formatted as type pathlib.Path()"

        print(f'Validation -- PASSED --')

    def _set_to_torch_set(self, transforms, num_ids):
        if transforms is None:
            transforms = []
        if not isinstance(transforms, list):
            transforms = [transforms]
        transforms.insert(0, AlignInputChannels(self._in_channels))
        transforms = Compose(transforms)
        return PointDatasetLoaderBridge(self, self._transformation_finalizer(transforms), loader_len=num_ids)

    # @abstractmethod
    def _transformation_finalizer(self, transforms):
        raise NotImplementedError

    # @abstractmethod
    def _hierarchical_index_to_path(self, hi):
        raise NotImplementedError

    # @abstractmethod
    def _path_load(self, fps):
        raise NotImplementedError

    # @abstractmethod
    def _construct_hit(self):
        raise NotImplementedError

    # @abstractmethod
    def show_sample(self, montage_shape):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class PointDatasetLoaderBridge(Dataset):
    # Note - This class is pretty hacky.
    # Note that changes to Dataset will be seen in any loader derived from it before
    # This should be taken into account when decimating the Dataset index
    def __init__(self, ds_inst, transforms, loader_len):
        self._ds_inst = ds_inst
        self._transforms = transforms
        # todo - attach dataset to Compose if dataset is SMPL - for immediate access to the faces() in transforms
        # It's faster - but requires thought if faces is unique
        self._loader_len = loader_len

    def __len__(self):
        return self._loader_len

    def __getitem__(self, si):
        hi = self._ds_inst._hit.si2hi(si)
        fp = self._ds_inst._hierarchical_index_to_path(hi)
        data = self._ds_inst._path_load(fp)
        return self._transforms(data)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionProjDataset(PointDataset, ABC):
    def __init__(self, data_dir_override, cls, shape, in_channels, disk_space_bytes, in_cfg):
        super().__init__(data_dir_override=data_dir_override, cls=cls, shape=shape,
                         in_channels=in_channels, disk_space_bytes=disk_space_bytes)
        self._in_cfg = in_cfg
        self._proj_dir = self._data_dir / 'projections'
        self._full_dir = self._data_dir / 'full'
        # Bind the methods at runtime:
        self._hierarchical_index_to_path = MethodType(getattr(self.__class__, f"_{in_cfg.name.lower()}_path"), self)
        self._path_load = MethodType(getattr(self.__class__, f"_{in_cfg.name.lower()}_load"), self)

        from cfg import DANGEROUS_MASK_THRESH, DEF_COMPUTE_PRECISION
        self._mask_thresh = DANGEROUS_MASK_THRESH
        self._def_precision = getattr(np, DEF_COMPUTE_PRECISION)

    def _transformation_finalizer(self, transforms):
        # A bit messy
        keys = [('gt_part', 'gt_mask_vi', 'gt')]
        if enum_eq(self._in_cfg, InCfg.PART2PART):
            keys.append(('tp', 'tp_mask_vi', 'tp'))  # Override tp
        transforms.append(PartCompiler(keys))
        return transforms

    def _full2part_path(self, hi):
        gt_fp = self._hi2full_path(hi)
        mask_fp = self._hi2proj_path(hi)
        # New index from the SAME subject
        tp_hi = self._hit.random_path_from_partial_path([hi[0]])
        tp_fp = self._hi2full_path(tp_hi)
        return [hi, gt_fp, mask_fp, tp_hi, tp_fp]  # TODO - Think about just using dicts all the way

    def _full2part_load(self, fps):
        # TODO - Add in support for faces that are loaded from file - by overloading hi2full for example

        gt_hi = fps[0]
        gt = self._full2data(fps[1]).astype(self._def_precision)
        gt_mask_vi = self._proj2data(fps[2])
        tp_hi = fps[3]
        tp = self._full2data(fps[4]).astype(self._def_precision)
        if len(gt_mask_vi) < self._mask_thresh:
            warn(f'Found mask of length {len(gt_mask_vi)} with id: {gt_hi}')

        return {'f': self._f, 'gt_hi': gt_hi, 'gt': gt, 'gt_mask_vi': gt_mask_vi, 'tp_hi': tp_hi, 'tp': tp}

    def _part2part_path(self, hi):
        fps = self._full2part_path(hi)
        # Use the old function and the new_hi to compute the part fp:
        fps.append(self._hi2proj_path(fps[3]))
        return fps

    def _part2part_load(self, fps):
        comp_d = self._full2part_load(fps)
        tp_mask_vi = self._proj2data(fps[5])
        if len(tp_mask_vi) < self._mask_thresh:
            warn(f'Found mask of length {len(tp_mask_vi)} with id: {comp_d["tp_hi"]}')
        comp_d['tp_mask_vi'] = tp_mask_vi
        return comp_d

    def show_sample(self, n_shapes=8, key='gt_part', strategy='spheres'):
        from util.mesh_visuals import plot_mesh_montage
        assert strategy in ['spheres', 'mesh', 'cloud']
        using_full = key in ['gt', 'tp']
        # assert not (not using_full and strategy == 'mesh'), "Mesh strategy for 'part' gets stuck in vtkplotter"
        fp_fun = self._hi2full_path if using_full else self._hi2proj_path
        samp = self.sample(n_shapes)

        origin = key.split('_')[0]
        labelb = [f'{key} | {fp_fun(samp[f"{origin}_hi"][i]).name}' for i in range(n_shapes)]
        vb = samp[key][:, :, 0:3].numpy()

        if strategy == 'mesh':
            if using_full:
                fb = self._f
            else:
                # TODO - Should we change the vertices as well?
                fb = [trunc_to_vertex_subset(vb[i], self._f, samp[f'{origin}_mask_vi'][i])[1] for i in range(n_shapes)]
        else:
            fb = None

        plot_mesh_montage(vb=vb, fb=fb, labelb=labelb, spheres_on=(strategy == 'spheres'),smooth_shade_on=True)

        # @abstractmethod

    def _hi2proj_path(self, hi):
        raise NotImplementedError

    # @abstractmethod
    def _hi2full_path(self, hi):
        raise NotImplementedError

    # @abstractmethod
    def _proj2data(self, fp):
        raise NotImplementedError

    # @abstractmethod
    def _full2data(self, fp):
        raise NotImplementedError


class SMPLCompletionProjDataset(CompletionProjDataset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add in SMPL faces
        with open(self._data_dir / 'SMPL_face.pkl', "rb") as f_file:
            self._f = load(f_file)

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
