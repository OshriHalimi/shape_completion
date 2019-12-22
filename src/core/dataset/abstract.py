from pathlib import Path
import os
from torch.utils.data import DataLoader
from torch._six import container_abcs, string_classes, int_classes
import torch.utils.data
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import MutableMapping
from torch.utils.data.sampler import SubsetRandomSampler
from bisect import bisect_right
from pprint import pprint
from util.container import max_dict_depth, min_dict_depth
from util.gen import banner, convert_bytes, warn
from util.nn import determine_worker_num
from collections import defaultdict
from pickle import dump, load
from dataset.transforms import *
from json import dumps
from copy import deepcopy
import time

# ----------------------------------------------------------------------------------------------------------------------
#                                             Static Variables
# ----------------------------------------------------------------------------------------------------------------------
PRIMARY_DATA_DIR = Path(__file__).parents[0] / '..' / '..' / '..' / 'data'
SUPPORTED_IN_CHANNELS = (3, 6, 12)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class PointDataset(ABC):
    def __init__(self, shape, disk_space_bytes, is_synthetic, existing_in_channels, num_disk_accesses,
                 in_channels, data_dir, validate):
        # Basic Dataset Info
        self._hit = self._construct_hit()
        self._shape = tuple(shape)
        self._disk_space_bytes = disk_space_bytes
        self._is_synthetic = is_synthetic
        self._existing_in_channels = existing_in_channels
        self._in_channels = in_channels
        self._num_disk_accesses = num_disk_accesses
        self._f = None
        # Strong assumption here - Only two kinds of datasets, synthetic and scan - bad coding.
        if data_dir is None:
            appender = 'synthetic' if is_synthetic else 'scan'
            self._data_dir = PRIMARY_DATA_DIR / appender / self.name()
        else:
            self._data_dir = Path(data_dir)
        assert self._data_dir.is_dir(), f"Data directory of {self.name()} is invalid: \nCould not find {self._data_dir}"
        assert in_channels in SUPPORTED_IN_CHANNELS  #
        self._data_dir = self._data_dir.resolve()

        if validate:
            self._validate_dataset()

    def data_summary(self, show_sample=False, with_tree=False):
        banner('Dataset Summary')
        print(f'* Dataset Name: {self.name()}')
        print(f'* Point Cloud Number: {self.num_pnt_clouds()}')
        print(f'* Input Data Shape: {self._shape}')
        print(f'* Number of Input Channels on Disk: {self._existing_in_channels}')
        print(f'* Number of Input Channels after Truncation: {self._in_channels}')
        print(f'* Number of Hard-disk accesses per shape: {self._num_disk_accesses}')
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
            # TODO - Add shape sanity check here
            return (train_loader, len(ids)), (valid_loader, len(valid_ids))
        else:
            train_loader = DataLoader(self._set_to_torch_set(transforms), batch_size=batch_size,
                                      sampler=SubsetRandomSampler(ids), num_workers=num_workers,
                                      pin_memory=pin_memory)
            return train_loader, len(ids)

    def _validate_dataset(self):
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


class CompletionProjDataset(PointDataset, ABC):
    def _transformation_finalizer(self, transforms):
        transforms.append(CompletionPairToTuple())  # Add in a final layer that destroys the CompletionPair
        return transforms


class SMPLCompletionProjDataset(CompletionProjDataset, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
#                                    Utility Methods - TODO - migrate to Data Prep
# ----------------------------------------------------------------------------------------------------------------------

def construct_dfaust_hit_pickel():
    import re
    dataset_fp = Path(PRIMARY_DATA_DIR) / 'synthetic' / 'DFaustPyProj'
    fp = dataset_fp / "dfaust_subjects_and_sequences.txt"
    assert fp.is_file(), "Could not find Dynamic Faust subjects_and_sequences.txt file"
    print(f'Found dfaust subjects and sequence list at {fp.parents[0].resolve()}')
    with open(fp) as f:
        lines = [line.rstrip('\n') for line in f]

    hit = defaultdict(dict)
    last_subj = None

    for line in lines:
        m = re.match(r"(\d+)\s+\((\w+)\)", line)
        if m:  # New hit
            sub_id, _ = m.group(1, 2)
            last_subj = sub_id
            hit[last_subj] = defaultdict(dict)
        elif line.strip():
            seq, frame_cnt = line.split()
            hit[last_subj][seq][int(frame_cnt)] = 10

    hit = HierarchicalIndexTree(hit, in_memory=True)

    pkl_fp = dataset_fp / "DFaust_hit.pkl"
    with open(pkl_fp, "wb") as f:
        dump(hit, f)
    print(f'Dumped Dynamic Faust HIT Pickle at {pkl_fp.resolve()}')
    return hit


def construct_amass_hit_pickels():

    import json
    dict_dir = Path(r'C:\Users\idoim\Desktop\ShapeCompletion\src\core\archive') # fix me as needed
    for name in ['test_dict.json','train_dict.json','vald_dict.json']:
        hit_name_appender = name.split('_')[0]
        fp = dict_dir / name
        with open(fp) as f:
            hit = json.loads(f.read())

        # Transform:
        ks = list(hit.keys())
        for k in ks:
            val = hit.pop(k)
            k = int(k)
            hit[k] = dict()
            for i in range(val):
                hit[k][i] = 10

        if name == 'test_dict.json':
            hit = HierarchicalIndexTree(hit,in_memory=True)
        else:
            hit = HierarchicalIndexTree(hit, in_memory=False)
        with open(dict_dir / f'amass_{hit_name_appender}_hit.pkl', "wb") as file:
          dump(hit,file)

        # print(hit)
        # print(hit.num_objects())

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Index Tree Mapping
# ----------------------------------------------------------------------------------------------------------------------
class HierarchicalIndexTree:
    def __init__(self, hit, in_memory):

        # Seems that dict() maintains insertion order by default from Py 3.7>,
        # and I'm presuming Python 3.7 is used to run this code
        # if not deep_check_odict(hit):
        # warn("Hierarchical Index Tree provided is not ordered - using sorted() order",stacklevel=2)
        # hit = deep_dict_to_odict(hit)
        if not hit:
            warn("Hierarchical Index Tree is empty", stacklevel=2)
        max_depth = max_dict_depth(hit)
        min_depth = min_dict_depth(hit)
        assert max_depth == min_depth, "Hierarchical Index Tree must be balanced"

        self._hit = hit
        self._in_memory = in_memory
        self._depth = max_depth
        self._multid_list, self._num_objs = self._flatten()  # multid has two possible forms, in dependence of in_memory

    def __str__(self):
        # pf = pformat(self._hit)
        # return pf.replace('OrderedDict', 'HierarchicalIndexTree')
        return 'HierarchicalIndexTree' + dumps(self._hit, indent=2)

    def depth(self):
        return self._depth

    def num_objects(self):
        return self._num_objs

    def get_id_union_by_depth(self, depth):
        if depth > self._depth - 1 or depth <= 0:
            # We don't allow getting of ids from the lowest level - seeing they are not unique
            raise AssertionError(f"Invalid depth requested. Valid inputs are: {list(range(1, self._depth))}")

        key_set = set()
        self._get_id_union_by_depth(self._hit, depth - 1, key_set)
        return sorted(list(key_set))

    def remove_ids_by_depth(self, goners, depth):
        if depth > self._depth - 1 or depth <= 0:
            # We don't allow getting of ids from the lowest level - seeing they are not unique
            raise AssertionError(f"Invalid depth requested. Valid inputs are: {list(range(1, self._depth))}")

        if type(goners) is not list:
            goners = [goners]
        reduced_hit = self._remove_ids_by_depth(self._hit, depth - 1, set(goners))
        return HierarchicalIndexTree(reduced_hit, self._in_memory)

    def si2hi(self, si):

        if si > self._num_objs:
            raise IndexError  # Total count
        if self._in_memory:  # Presuming extended flattend multindex list
            return self._multid_list[si]
        else:  # Presuming accumulation list structure
            tgt_bin_id = bisect_right(self._multid_list[0], si) - 1
            multilevel_id = list(self._multid_list[1][tgt_bin_id])
            acc_cnt_for_bin_id = self._multid_list[0][tgt_bin_id]
            multilevel_id.append(si - acc_cnt_for_bin_id)
            return tuple(multilevel_id)

    def _flatten(self):

        flat_hit = self._flatten_aux(self._hit, self._in_memory)
        flat_hit = [tuple(o) for o in flat_hit]  # Turn into tuple for correctness
        flat_hit = tuple(flat_hit)

        if self._in_memory:
            return flat_hit, len(flat_hit)
        else:
            acc_list = [[], []]  # Accumulation list, multi-index per bin
            tot_obj_cnt = 0
            for multindex in flat_hit:
                acc_list[0].append(tot_obj_cnt)
                acc_list[1].append(multindex[0:len(multindex) - 1])
                tot_obj_cnt += multindex[
                    -1]  # Increase the accumulation by the last member in multilevel index tree - the object cnt
            return acc_list, tot_obj_cnt

    @staticmethod
    def _get_id_union_by_depth(hit, depth, dump):
        if depth == 0:
            dump |= set(hit.keys())
        else:
            for v in hit.values():
                HierarchicalIndexTree._get_id_union_by_depth(v, depth - 1, dump)

    @staticmethod
    def _remove_ids_by_depth(hit, depth, goners):
        modified_dict = OrderedDict()
        for key, value in hit.items():
            if depth != 0 or key not in goners:  # Keep it
                if isinstance(value, MutableMapping):
                    child_dict = HierarchicalIndexTree._remove_ids_by_depth(value, depth - 1, goners)
                    if child_dict:  # Assert it is not empty
                        modified_dict[key] = child_dict
                else:
                    modified_dict[key] = value
        return modified_dict

    @staticmethod
    def _flatten_aux(hit, extended_flatten):
        if isinstance(hit, MutableMapping):
            local_list = []  # Holder of all the sublists
            for key, value in hit.items():
                child_lists = HierarchicalIndexTree._flatten_aux(value, extended_flatten)
                for l in child_lists:
                    l.insert(0, key)
                local_list += child_lists  # Append children's lists:
            return local_list
        else:  # is value
            if extended_flatten:
                assert isinstance(hit, int) and hit > 0  # Checks validity of structure
                return [[i] for i in range(hit)]
            else:
                return [[hit]]


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Modules
# ----------------------------------------------------------------------------------------------------------------------

def hit_test():
    print = (lambda p: lambda *args, **kwargs: [p(*args, **kwargs), time.sleep(.01)])(print)
    hit = {'Subject1':
               {'Pose1': 200,
                'Pose2': 300,
                # 'Pose3': {
                #     'Seq1': 500,
                #     'Seq2': 600
                # }
                },
           'Subject2':
               {'Pose1': 1,
                'Pose2': 4,
                'Pose3': 4,
                },
           'Subject3':
               {'Pose1': 100,
                'Pose2': 50,
                },
           # 'Subject 4': 3
           }

    hit_mem = HierarchicalIndexTree(hit, in_memory=True)
    hit_out_mem = HierarchicalIndexTree(hit, in_memory=False)

    banner('In Memory vs Out of Memory Tests')
    print(hit_mem)
    print(hit_mem.depth())
    print(hit_mem.num_objects())
    print(hit_out_mem)
    print(hit_out_mem.depth())
    print(hit_out_mem.num_objects())

    ids = range(hit_mem.num_objects())
    for i in ids:
        # print(hit_mem.si2hi(i))
        # print(hit_out_mem.si2hi(i))
        assert (hit_mem.si2hi(i) == hit_out_mem.si2hi(i))

    banner('ID Union Tests')
    pprint(hit_mem.get_id_union_by_depth(depth=1))
    pprint(hit_mem.get_id_union_by_depth(depth=2))
    banner('Removal Tests')
    print(hit_mem.remove_ids_by_depth('Subject1', depth=1))
    print(hit_mem.remove_ids_by_depth(['Subject1', 'Subject2'], 1))
    print(hit_mem.remove_ids_by_depth(['Subject1', 'Subject2', 'Subject3'], 1))
    print(hit_mem.remove_ids_by_depth(['Pose1'], 2))
    print(hit_mem.remove_ids_by_depth(['Pose1', 'Pose2'], 2))
    print(hit_mem.remove_ids_by_depth(['Pose1', 'Pose2', 'Pose3'], 2))


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # hit = construct_dfaust_hit_pickel()
    # print(hit)
    construct_amass_hit_pickels()
    # hit_test()

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
# fp = r'C:\Users\idoim\Desktop\shape_completion\data\synthetic\FaustPyProj\full\tr_reg_000.off'
# from util.mesh_file import read_off_full
# _,f = read_off_full(fp)
# with open("SMPL_face.pkl", "wb") as file:
#   dump(f,file)
