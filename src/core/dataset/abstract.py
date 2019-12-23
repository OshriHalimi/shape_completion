from pathlib import Path
import os
from torch.utils.data import DataLoader
import torch.utils.data
from abc import ABC, abstractmethod
from torch.utils.data.sampler import SubsetRandomSampler
from util.gen import banner, convert_bytes
from util.nn import determine_worker_num
from pickle import load
from copy import deepcopy
from dataset.transforms import *
from enum import Enum, auto
from tqdm import tqdm
import time

# ----------------------------------------------------------------------------------------------------------------------
#                                             Global Variables
# ----------------------------------------------------------------------------------------------------------------------
PRIMARY_DATA_DIR = Path(__file__).parents[0] / '..' / '..' / '..' / 'data'
SUPPORTED_IN_CHANNELS = (3, 6, 12)


class InCfg(Enum):
    FULL_FULL_PART = auto()
    FULL_PART_PART = auto()


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class PointDataset(ABC):
    def __init__(self, data_dir_override, is_synthetic, shape, in_channels, disk_space_bytes):
        # Check Data Directory:
        if data_dir_override is None:
            # TODO - Remove assumption of only synthetic and scan
            appender = 'synthetic' if is_synthetic else 'scan'
            self._data_dir = PRIMARY_DATA_DIR / appender / self.name()
        else:
            self._data_dir = Path(data_dir_override)
        assert self._data_dir.is_dir(), f"Data directory of {self.name()} is invalid: \nCould not find {self._data_dir}"
        self._data_dir = self._data_dir.resolve()

        # Basic Dataset Info
        self._hit = self._construct_hit()

        # Insert Info:
        self._shape = tuple(shape)
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
        if show_sample:  # TODO - Not implemented
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
            return torch.from_numpy(self._f).cuda()  # TODO - Why was long() here?
        else:
            return deepcopy(self._f)

    def testloader(self, batch_size=16, set_size=None, shuffle=False, device='cuda'):
        return self._loader(batch_size=batch_size, vald_rsize=0, set_size=set_size, transforms=None, shuffle=shuffle,
                            device=device)

    def trainloader(self, batch_size=16, vald_rsize=0.1, set_size=None, transforms=None, shuffle=True, device='cuda'):
        return self._loader(batch_size=batch_size, vald_rsize=vald_rsize, set_size=set_size, transforms=transforms,
                            shuffle=shuffle, device=device)

    def _loader(self, batch_size, vald_rsize, set_size, transforms, shuffle, device):

        # Handle Loader Parameters
        num_workers = determine_worker_num(batch_size)
        pin_memory = (torch.cuda.is_available() and device.lower() == 'cuda')
        set_size = self.num_pnt_clouds() if set_size is None else set_size
        max_samples = min(self.num_pnt_clouds(), set_size)
        if max_samples < set_size:
            warn(f'Set size: {set_size} requested, while dataset holds: {self.num_pnt_clouds()}. Reverting to latter.')

        # Handle Set Size and Shuffle
        ids = list(range(self.num_pnt_clouds()))
        if shuffle:
            np.random.shuffle(ids)
        ids = ids[:max_samples]  # Shuffles sets may now hold different ids than 1:max_samples

        # Split validation
        assert ((vald_rsize >= 0) and (vald_rsize <= 1)), "[!] Valid_size should be in the range [0, 1]."
        if vald_rsize > 0:
            split = int(np.floor(vald_rsize * max_samples))
            ids, vald_ids = ids[split:], ids[:split]
            train_loader = DataLoader(self._set_to_torch_set(transforms), batch_size=batch_size,
                                      sampler=SubsetRandomSampler(ids), num_workers=num_workers,
                                      pin_memory=pin_memory)
            vald_loader = DataLoader(self._set_to_torch_set(None), batch_size=batch_size,
                                     sampler=SubsetRandomSampler(vald_ids), num_workers=num_workers,
                                     pin_memory=pin_memory)
            # Add shape sanity check here ?
            return (train_loader, len(ids)), (vald_loader, len(vald_ids))
        else:
            train_loader = DataLoader(self._set_to_torch_set(transforms), batch_size=batch_size,
                                      sampler=SubsetRandomSampler(ids), num_workers=num_workers,
                                      pin_memory=pin_memory)
            return train_loader, len(ids)

    def validate_dataset(self):
        banner(f'Validation of dataset {self.name()} :: {self.num_pnt_clouds()} pnt clouds')
        time.sleep(.01)  # For the STD-ERR lag
        for si in tqdm(range(self.num_pnt_clouds())):
            hi = self._hit.si2hi(si)
            fps = self._hierarchical_index_to_path(hi)
            if not isinstance(fps, list):
                fps = [fps]
            # TODO - Maybe add a count of possible missing files? Not really needed, seeing working on a partial dataset
            # requires updating the hit
            # TODO - Note that it is not really needed to iterate over all the fps - only the projections + full set is
            # enough - Better to change the call to something different maybe?
            for fp in fps:
                if isinstance(fp, Path):
                    assert fp.is_file(), f"Missing file {fp.resolve()} in dataset {self.name()}"
                else:
                    try:
                        fp = Path(fp)
                        assert fp.isfile(), f"Missing file {fp.resolve()} in dataset {self.name()}"
                    except:
                        pass  # Some none-path object hidden in fp

        print(f'Validation -- PASSED --')

    def _show_sample(self, montage_shape):  # TODO - Finish this
        raise NotImplementedError
        # images, labels = iter(torch.util.data.DataLoader(train_dataset, shuffle=True, batch_size=siz ** 2)).next()
        # plot_images(images.numpy().transpose([0, 2, 3, 1]), labels, self._class_labels, siz=siz)

    def _set_to_torch_set(self, transforms):
        if transforms is None:
            transforms = []
        transforms.insert(0, AlignInputChannels(self._in_channels))
        transforms = Compose(transforms)
        return PointDatasetLoaderBridge(self, self._transformation_finalizer(transforms))

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
#
# ----------------------------------------------------------------------------------------------------------------------
class CompletionProjDataset(PointDataset, ABC):
    def __init__(self, data_dir_override, is_synthetic, shape, in_channels, disk_space_bytes, in_cfg):
        super().__init__(data_dir_override=data_dir_override, is_synthetic=is_synthetic, shape=shape,
                         in_channels=in_channels, disk_space_bytes=disk_space_bytes)
        self._in_cfg = in_cfg
        self._proj_dir = self._data_dir / 'projections'
        self._full_dir = self._data_dir / 'full'
        self._path_meth = getattr(self.__class__, f"_{in_cfg.name.lower()}_path")
        self._data_load_meth = getattr(self.__class__, f"_{in_cfg.name.lower()}_load")

    def _transformation_finalizer(self, transforms):
        transforms.append(CompletionTripletToTuple())  # Add in a final layer that dismantles the CompletionTriplet
        return transforms

    def _full_full_part_path(self, hi):
        gt_fp = self._hi2full_path(hi)
        mask_fp = self._hi2proj_path(hi)
        # New index from the SAME subject
        new_hi = self._hit.random_path_from_partial_path([hi[0]])
        tp_fp = self._hi2full_path(new_hi)
        return [hi, gt_fp, mask_fp, new_hi, tp_fp]

    def _full_full_part_load(self, fps):
        # TODO - Add in support for faces that are loaded from file - by overloading hi2full for example
        return CompletionTriplet(f=self._f, hi=fps[0], gt_v=self._full2data(fps[1]), mask_vi=self._proj2data(fps[2]),
                                 new_hi=fps[3], tp_v=self._full2data(fps[4]))

    def _full_part_part_path(self, hi):
        fps = self._full_full_part_path(hi)
        # Use the old function and the new_hi to compute the part fp:
        fps.append(self._hi2proj_path(fps[3]))
        return fps

    def _full_part_part_load(self, fps):
        ct = self._full_full_part_load(fps)
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

class PointDatasetLoaderBridge(torch.utils.data.Dataset):
    # TODO - This class is pretty hacky. Potentially undesired effects when Loader detaches from Dataset
    def __init__(self, ds_inst, transforms):
        self._ds_inst = ds_inst
        self._transforms = transforms

    def __len__(self):
        return len(self._ds_inst.num_pnt_clouds())

    def __getitem__(self, si):
        hi = self._ds_inst._hit.si2hi(si)
        fp = self._ds_inst._hierarchical_index_to_path(hi)
        data = self._ds_inst._path_load(fp)
        return self._transforms(data)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def abstract_test():
    x = InCfg.FULL_FULL_PART
    print(x.name.lower())


if __name__ == "__main__":
    abstract_test()
