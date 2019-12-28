from pathlib import Path
from torch.utils.data import DataLoader
from dataset.collate import default_collate
import torch.utils.data
from abc import ABC, abstractmethod
from torch.utils.data.sampler import SubsetRandomSampler
from util.gen import banner, convert_bytes
from util.container import split_frac
from pickle import load
from copy import deepcopy
from dataset.transforms import *
from enum import Enum, auto
from tqdm import tqdm
import time
import psutil

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

        # Basic Dataset Info
        self._hit = self._construct_hit()

        # Insert Info:
        self._shape = tuple(shape)
        from cfg import SUPPORTED_IN_CHANNELS
        assert in_channels in SUPPORTED_IN_CHANNELS
        self._in_channels = in_channels
        self._disk_space_bytes = disk_space_bytes

        self._f = None

    def data_summary(self, show_sample=False, with_tree=False):
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
        if show_sample:
            self._show_sample(montage_shape=(3, 3))

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

    def faces(self, torch_version=False):
        assert self._f is not None, "Faces property is empty"
        if torch_version:
            # GPU doesn't deal well with int32 versions of the data -> That's why we transfer it to long
            return torch.from_numpy(self._f).long().cuda(), deepcopy(self._f)  # TODO - Make normal compute all GPU
        else:
            return deepcopy(self._f)

    def loader(self, ids=None, transforms=None, batch_size=16, device='cuda'):
        if ids is None:
            ids = range(self.num_pnt_clouds())
        assert len(ids) > 0, "Found loader with no data samples inside"
        num_workers = determine_worker_num(batch_size)
        device = device.lower()
        assert device in ['cuda', 'cpu']
        pin_memory = (device == 'cuda')
        train_loader = DataLoader(self._set_to_torch_set(transforms, len(ids)), batch_size=batch_size,
                                  sampler=SubsetRandomSampler(ids), num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=default_collate)
        return train_loader

    def split_loaders(self, split, s_nums, s_shuffle, s_transform, global_shuffle=False, batch_size=16, device='cuda'):
        """
        # s for split
        :param split: A list of fracs summing to 1: e.g.: [0.9,0.1] or [0.8,0.15,0.05]
        :param s_nums: A list of integers: e.g. [1000,50] or [1000,5000,None] - The number of objects to take from each
        range split. If None, it will take the maximal number possible
        :param s_shuffle: A list of booleans: If s_shuffle[i]==True, the ith split will be shuffled before truncations
        to s_nums[i] objects
        :param s_transform: A list - s_transforms[i] is the transforms for the ith split
        :param global_shuffle: If True, shuffles the entire set before split
        :param batch_size: Integer > 0
        :param device: 'cuda' or 'cpu'
        :return: A list of (loaders,num_samples)
        """

        assert sum(split) == 1, "Split fracs must sum to 1"
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
            loaders.append(self.loader(ids=set_ids, transforms=transforms, batch_size=batch_size, device=device))

        if n_parts == 1:
            loaders = loaders[0]
        return loaders

    def validate_dataset(self):
        banner(f'Validation of dataset {self.name()} :: {self.num_pnt_clouds()} pnt clouds')
        time.sleep(.01)  # For the STD-ERR lag
        for si in tqdm(range(self.num_pnt_clouds())):
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

    def _show_sample(self, montage_shape):  # TODO - Not implemented
        raise NotImplementedError
        # images, labels = iter(torch.util.data.DataLoader(train_dataset, shuffle=True, batch_size=siz ** 2)).next()
        # plot_images(images.numpy().transpose([0, 2, 3, 1]), labels, self._class_labels, siz=siz)

    def _set_to_torch_set(self, transforms, num_ids):
        if transforms is None:
            transforms = []
        if not isinstance(transforms, list):
            transforms = [transforms]
        transforms.insert(0, AlignInputChannels(self._in_channels))
        transforms = Compose(transforms)
        return PointDatasetLoaderBridge(self, self._transformation_finalizer(transforms), loader_len=num_ids)

    @abstractmethod
    def _transformation_finalizer(self, transforms):
        raise NotImplementedError

    @abstractmethod
    def _hierarchical_index_to_path(self, hi):
        raise NotImplementedError

    @abstractmethod
    def _path_load(self, fps):
        raise NotImplementedError

    @abstractmethod
    def _construct_hit(self):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class PointDatasetLoaderBridge(torch.utils.data.Dataset):
    # Note - This class is pretty hacky.
    # Note that changes to Dataset will be seen in any loader derived from it before
    # This should be taken into account when decimating the Dataset index
    def __init__(self, ds_inst, transforms, loader_len):
        self._ds_inst = ds_inst
        self._transforms = transforms
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

def loader_ids(loader):
    return list(loader.batch_sampler.sampler.indices)


def exact_num_loader_obj(loader):
    # This is more exact than len(loader)*batch_size - Seeing we don't round up the last batch to batch_size
    return len(loader.dataset)

def determine_worker_num(batch_size):
    cpu_cnt = psutil.cpu_count(logical=False)
    if batch_size < cpu_cnt:
        return batch_size
    else:
        return cpu_cnt

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
        self._path_meth = getattr(self.__class__, f"_{in_cfg.name.lower()}_path")
        self._data_load_meth = getattr(self.__class__, f"_{in_cfg.name.lower()}_load")

    def _transformation_finalizer(self, transforms):
        transforms.append(CompletionTripletToTuple())  # Add in a final layer that dismantles the CompletionTriplet
        return transforms

    def _full2part_path(self, hi):
        gt_fp = self._hi2full_path(hi)
        mask_fp = self._hi2proj_path(hi)
        # New index from the SAME subject
        new_hi = self._hit.random_path_from_partial_path([hi[0]])
        tp_fp = self._hi2full_path(new_hi)
        return [hi, gt_fp, mask_fp, new_hi, tp_fp]

    def _full2part_load(self, fps):
        # TODO - Add in support for faces that are loaded from file - by overloading hi2full for example
        return CompletionTriplet(f=self._f, hi=fps[0], gt_v=self._full2data(fps[1]), mask_vi=self._proj2data(fps[2]),
                                 new_hi=fps[3], tp_v=self._full2data(fps[4]))

    def _part2part_path(self, hi):
        fps = self._full2part_path(hi)
        # Use the old function and the new_hi to compute the part fp:
        fps.append(self._hi2proj_path(fps[3]))
        return fps

    def _part2part_load(self, fps):
        ct = self._full2part_load(fps)
        ct.tp_mask_vi = self._proj2data(fps[5])
        return ct

    def _hierarchical_index_to_path(self, hi):
        return self._path_meth(self, hi)

    def _path_load(self, fps):
        return self._data_load_meth(self, fps)

    @abstractmethod
    def _hi2proj_path(self, hi):
        raise NotImplementedError

    @abstractmethod
    def _hi2full_path(self, hi):
        raise NotImplementedError

    @abstractmethod
    def _proj2data(self, fp):
        raise NotImplementedError

    @abstractmethod
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

if __name__ == "__main__":
    pass

# ----------------------------------------------------------------------------------------------------------------------
#                                                     Graveyard
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
