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
    def __init__(self, data_dir, is_synthetic, shape, in_channels, disk_space_bytes):
        # Basic Dataset Info
        self._hit = self._construct_hit()

        # Check Data Directory:
        if data_dir is None:
            # TODO - Remove assumption of only synthetic and scan
            appender = 'synthetic' if is_synthetic else 'scan'
            self._data_dir = PRIMARY_DATA_DIR / appender / self.name()
        else:
            self._data_dir = Path(data_dir)
        assert self._data_dir.is_dir(), f"Data directory of {self.name()} is invalid: \nCould not find {self._data_dir}"
        self._data_dir = self._data_dir.resolve()

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
        return self._loader(batch_size=batch_size, valid_rsize=0, set_size=set_size, transforms=None, shuffle=shuffle,
                            device=device)

    def trainloader(self, batch_size=16, valid_rsize=0.1, set_size=None, transforms=None, shuffle=True, device='cuda'):
        return self._loader(batch_size=batch_size, valid_rsize=valid_rsize, set_size=set_size, transforms=transforms,
                            shuffle=shuffle, device=device)

    def _loader(self, batch_size, valid_rsize, set_size, transforms, shuffle, device):

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
        assert ((valid_rsize >= 0) and (valid_rsize <= 1)), "[!] Valid_size should be in the range [0, 1]."
        if valid_rsize > 0:
            split = int(np.floor(valid_rsize * max_samples))
            ids, valid_ids = ids[split:], ids[:split]
            train_loader = DataLoader(self._set_to_torch_set(transforms), batch_size=batch_size,
                                      sampler=SubsetRandomSampler(ids), num_workers=num_workers,
                                      pin_memory=pin_memory)
            valid_loader = DataLoader(self._set_to_torch_set(None), batch_size=batch_size,
                                      sampler=SubsetRandomSampler(valid_ids), num_workers=num_workers,
                                      pin_memory=pin_memory)
            # Add shape sanity check here ?
            return (train_loader, len(ids)), (valid_loader, len(valid_ids))
        else:
            train_loader = DataLoader(self._set_to_torch_set(transforms), batch_size=batch_size,
                                      sampler=SubsetRandomSampler(ids), num_workers=num_workers,
                                      pin_memory=pin_memory)
            return train_loader, len(ids)

    def validate_dataset(self):
        for si in range(self.num_pnt_clouds()):
            hi = self._hit.si2hi(si)
            fps = self._hierarchical_index_to_path(hi)
            if not isinstance(fps, list):
                fps = [fps]
            for fp in fps:
                assert os.path.isfile(fp), f"Missing file {fp} in dataset {self.name()}"

        print(f'Validation for {self.name()} passed')

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
    def __init__(self, data_dir, is_synthetic, shape, in_channels, disk_space_bytes, in_cfg):
        super().__init__(data_dir=data_dir, is_synthetic=is_synthetic, shape=shape,
                         in_channels=in_channels, disk_space_bytes=disk_space_bytes)
        self._in_cfg = in_cfg
        self._proj_dir = self._data_dir / 'projections'
        self._full_dir = self._full_dir / 'full'
        self._path_meth = getattr(self.__class__, f"_{in_cfg.name}_path")
        self._data_load_meth = getattr(self.__class__, f"_{in_cfg.name}_load")

    def _transformation_finalizer(self, transforms):
        transforms.append(CompletionPairToTuple())  # Add in a final layer that destroys the CompletionPair
        return transforms

    def full_full_part_path(self, hi):
        pass

    def full_full_part_load(self, fps):
        pass

    def full_part_part_path(self, hi):
        pass

    def full_part_part_load(self, fps):
        pass

    def _hierarchical_index_to_path(self, hi):
        return self._path_meth(hi)

    def _path_load(self, fps):
        return self._data_load_meth(fps)

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
