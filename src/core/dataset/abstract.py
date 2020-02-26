from torch.utils.data import Dataset
from pathlib import Path
from abc import ABC
from util.torch.data import determine_worker_num, ReconstructableDataLoader, ParametricLoader, SubsetChoiceSampler
from util.strings import warn, banner
from util.func import time_me
from util.fs import convert_bytes
from util.container import split_frac, to_list
from util.mesh.plots import plot_mesh
from pickle import load
from dataset.transforms import *
from tqdm import tqdm
import sys
import torch
import re
from torch._six import container_abcs, string_classes, int_classes
from types import MethodType


# from torch.utils.data.distributed import DistributedSampler
# from mesh.ops import trunc_to_vertex_mask

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class HitIndexedDataset(ABC):
    def __init__(self, data_dir_override, cls, disk_space_bytes, suspected_corrupt=False):

        # Append Info:
        self._disk_space_bytes = disk_space_bytes
        self._suspected_corrupt = suspected_corrupt

        # Check Data Directory:
        if data_dir_override is None:
            from cfg import PRIMARY_DATA_DIR
            self._data_dir = PRIMARY_DATA_DIR / cls / self.name()
        else:
            self._data_dir = Path(data_dir_override)
        assert self._data_dir.is_dir(), f"Data directory of {self.name()} is invalid: \nCould not find {self._data_dir}"
        self._data_dir = self._data_dir.resolve()

        # Construct the hit
        self._hit = self._construct_hit()

    def data_summary(self, with_tree=False):
        banner('Dataset Summary')
        print(f'* Dataset Name: {self.name()}')
        print(f'* Number of Indexed Elements: {self.num_indexed()}')
        print(f'* Estimated Hard-disk space required: ~{convert_bytes(self._disk_space_bytes)}')
        print(f'* Direct Filepath: {self._data_dir}')
        if with_tree:
            banner('Dataset Index Tree')
            self.report_index_tree()
        banner()

    def report_index_tree(self):
        print(self._hit)

    def name(self):
        return self.__class__.__name__

    def num_indexed(self):
        return self._hit.num_indexed()

    def disk_space(self):
        return self._disk_space_bytes

    def validate_dataset(self):
        raise NotImplementedError

    def sample(self, num_samples):
        raise NotImplementedError

    def show_sample(self, num_samples):
        raise NotImplementedError

    def loaders(self):
        raise NotImplementedError

    def _construct_hit(self):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class FullPartCompletionDataset(HitIndexedDataset, ABC):
    DEFINED_SAMP_METHODS = ('full', 'part', 'f2p', 'rand_f2p', 'frand_f2p', 'p2p', 'rand_p2p', 'frand_p2p', 'rand_ff2p'
                            , 'rand_ff2pp', 'rand_f2p_seq')

    @classmethod
    def defined_methods(cls):
        return cls.DEFINED_SAMP_METHODS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._proj_dir = self._data_dir / 'projections'
        self._full_dir = self._data_dir / 'full'
        self._loader_cls = ReconstructableDataLoader
        self._f, self._n_v = None, None  # Place holder for same-face/same number of vertices - dataset

        from cfg import DANGEROUS_MASK_THRESH, UNIVERSAL_PRECISION
        self._def_precision = getattr(np, UNIVERSAL_PRECISION)
        self._mask_thresh = DANGEROUS_MASK_THRESH

        self._hit.init_cluster_hi_list()
        self._hit_in_memory = self._hit.in_memory()
        # TODO - Revise index map for datasets that are not in-memory
        self._tup_index_map = None

    def num_indexed(self):  # Override
        return self.num_projections() + self.num_full_shapes()

    def num_projections(self):
        return self._hit.num_indexed()

    def num_full_shapes(self):
        return self._hit.num_index_clusters()

    def num_datapoints_by_method(self, method):
        assert method in self.DEFINED_SAMP_METHODS
        if method == 'full':
            return self.num_full_shapes()
        elif method == 'part':
            return self.num_projections()
        elif method == 'f2p' or method == 'p2p':
            assert self._hit_in_memory, "Full tuple indexing will take too much time"  # TODO - Can this be fixed?
            if self._tup_index_map is None:
                self._build_tupled_index()  # TODO - Revise this for P2P
            return len(self._tup_index_map)
        else:
            return self.num_projections()  # This is big enough, but still a lie

    def data_summary(self, with_tree=False):
        v_str = "No" if self._n_v is None else f"Yes :: {self._n_v} vertices"
        f_str = "No" if self._f is None else f"Yes :: {self._f.shape[0]} faces"

        banner('Full-Part Dataset Summary')
        print(f'* Dataset Name: {self.name()}')
        print(f'* Index in memory: {"Yes" if self._hit_in_memory else "No"}')
        print(f'* Number of Full Shapes: {self.num_full_shapes()}')
        print(f'* Number of Projections {self.num_projections()}')
        print(f'* Uniform Face Array: {f_str}')
        print(f'* Uniform Number of Vertices: {v_str}')
        print(f'* Estimated Hard-disk space required: ~{convert_bytes(self._disk_space_bytes)}')
        print(f'* Direct Filepath: {self._data_dir}')
        if with_tree:
            banner('Dataset Index Tree')
            self.report_index_tree()
        banner()

    @time_me
    def validate_dataset(self):
        banner(f'Validation of dataset {self.name()} :: {self.num_projections()} Projections '
               f':: {self.num_full_shapes()} Full Shapes')
        for si in tqdm(range(self.num_projections()), file=sys.stdout, dynamic_ncols=True):
            hi = self._hit.si2hi(si)
            fp = self._hi2proj_path(hi)
            assert fp.is_file(), f"Missing projection {fp.resolve()} in dataset {self.name()}"
        for si in tqdm(range(self.num_full_shapes()), file=sys.stdout, dynamic_ncols=True):
            chi = self._hit.csi2chi(si)
            fp = self._hi2full_path(chi)
            assert fp.is_file(), f"Missing full subject {fp.resolve()} in dataset {self.name()}"
        print(f'Validation -- PASSED --')

    def sample(self, num_samples=10, transforms=None, method='f2p', n_channels=6):
        total_points = self.num_datapoints_by_method(method)
        if num_samples > total_points:
            warn(f"Requested {num_samples} samples when dataset only holds {total_points}. "
                 f"Returning the latter")
        ldr = self._loader(ids=None, batch_size=num_samples, device='cpu-single',
                           transforms=transforms, method=method, n_channels=n_channels, set_size=None)
        return next(iter(ldr))

    def show_sample(self, num_samples=4, strategy='mesh', with_vnormals=False, method='f2p'):
        raise NotImplementedError  # TODO

    def rand_loader(self, num_samples=None, transforms=(Center(),), batch_size=16, n_channels=6, mode='f2p',
                    device='cuda'):
        """
        :param num_samples: Number of samples in the dataloader, drawn at random from the entire dataset.
        This draw is down subject first->pose-> etc
        :param transforms: A list of the transforms you'd like to add, or None
        :param batch_size: The batch size
        :param n_channels: The number of input channels
        :param mode: Either f2p or p2p
        :param device: Insert a torch device/the string 'cuda','cpu' or 'cpu-single'
        :return: The dataloader
        """
        assert mode in ['f2p', 'p2p'], "Must choose one of ['f2p','p2p']"
        return self.loaders(s_nums=num_samples, s_transform=transforms, batch_size=batch_size, n_channels=n_channels,
                            method=f'frand_{mode}', device=device)

    def loaders(self, s_nums=None, s_shuffle=True, s_transform=None, split=(1,), s_dynamic=False,
                global_shuffle=False, batch_size=16, device='cuda', method='f2p', n_channels=6):
        """
        # s for split
        :param split: A list of fracs summing to 1: e.g.: [0.9,0.1] or [0.8,0.15,0.05]. Don't specify anything for a
        single loader
        :param s_nums: A list of integers: e.g. [1000,50] or [1000,5000,None] - The number of objects to take from each
        range split. If None, it will take the maximal number possible.
        WARNING: Remember that the data loader will ONLY load from these examples unless s_dynamic[i] == True
        :param s_dynamic: A list of booleans: If s_dynamic[i]==True, the ith split will take s_nums[i] examples at
        random from the partition [which usually includes many more examples]. On the next load, will take another
        random s_nums[i] from the partition. If False - will take always the very same examples. Usually, we'd want
        s_dynamic to be True only for the training set.
        :param s_shuffle: A list of booleans: If s_shuffle[i]==True, the ith split will be shuffled before truncations
        to s_nums[i] objects
        :param s_transform: A list - s_transforms[i] is the transforms for the ith split
        :param global_shuffle: If True, shuffles the entire set before split
        :param batch_size: Integer > 0
        :param device: 'cuda' or 'cpu' or 'cpu-single' or pytorch device
        :param method: One of ('full', 'part', 'f2p', 'rand_f2p','frand_f2p', 'p2p', 'rand_p2p','frand_p2p')
        :param n_channels: One of cfg.SUPPORTED_IN_CHANNELS - The number of channels required per datapoint
        :return: A list of (loaders,num_samples)
        """
        # Handle inpput arguments:
        s_shuffle = to_list(s_shuffle)
        s_dynamic = to_list(s_dynamic)
        s_nums = to_list(s_nums)
        if s_transform is None or not s_transform:
            s_transform = [None] * len(split)
            # Transforms must be a list, all others are non-Sequence
        assert sum(split) == 1, "Split fracs must sum to 1"
        # TODO - Clean up this function, add in smarter defaults, simplify
        if (method == 'f2p' or method == 'p2p') and not self._hit_in_memory:
            method = 'rand_' + method
            warn(f'Tuple dataset index is too big for this dataset. Reverting to {method} instead')
        if (method == 'frand_f2p' or method == 'frand_p2p') and len(split) != 1:
            raise ValueError("Seeing the fully-rand methods have no connection to the partition, we cannot support "
                             "a split dataset here")
        # Logic:
        ids = list(range(self.num_datapoints_by_method(method)))
        if global_shuffle:
            np.random.shuffle(ids)  # Mixes up the whole set

        n_parts = len(split)
        ids = split_frac(ids, split)
        loaders = []
        for i in range(n_parts):
            set_ids, req_set_size, do_shuffle, transforms, is_dynamic = ids[i], s_nums[i], \
                                                                        s_shuffle[i], s_transform[i], s_dynamic[i]
            if req_set_size is None:
                req_set_size = len(set_ids)
            eff_set_size = min(len(set_ids), req_set_size)
            if eff_set_size != req_set_size:
                warn(f'At Loader {i + 1}/{n_parts}: Requested {req_set_size} objects while set has only {eff_set_size}.'
                     f' Reverting to latter')
            if do_shuffle:
                np.random.shuffle(set_ids)  # Truncated sets may now hold different ids
            if not is_dynamic:  # Truncate only if not dynamic
                set_ids = set_ids[:eff_set_size]  # Truncate
            recon_stats = {
                'dataset_name': self.name(),
                'batch_size': batch_size,
                'split': split,
                'id_in_split': i,
                'set_size': eff_set_size,
                'transforms': str(transforms),
                'global_shuffle': global_shuffle,
                'partition_shuffle': do_shuffle,
                'method': method,
                'n_channels': n_channels,
                'in_memory_index': self._hit_in_memory,
                'is_dynamic': is_dynamic
            }

            ldr = self._loader(method=method, transforms=transforms, n_channels=n_channels, ids=set_ids,
                               batch_size=batch_size, device=device, set_size=eff_set_size)
            ldr.init_recon_table(recon_stats)
            loaders.append(ldr)

        if n_parts == 1:
            loaders = loaders[0]
        return loaders

    def _loader(self, method, transforms, n_channels, ids, batch_size, device, set_size):

        # Handle Device:
        device = str(device).split(':')[0]  # Compatible for both strings & pytorch devs
        assert device in ['cuda', 'cpu', 'cpu-single']
        pin_memory = (device == 'cuda')
        if device == 'cpu-single':
            n_workers = 0
        else:
            n_workers = determine_worker_num(len(ids), batch_size)

        # Compile Sampler:
        if ids is None:
            ids = range(self.num_datapoints_by_method(method))
        if set_size is None:
            set_size = len(ids)
        assert len(ids) > 0, "Found loader with no data samples inside"
        sampler_length = min(set_size, len(ids))  # Allows for dynamic partitions
        # if device == 'ddp': #TODO - Add distributed support here. What does num_workers need to be?
        # data_sampler == DistributedSampler(dataset,num_replicas=self.num_gpus,ranks=self.logger.rank)
        data_sampler = SubsetChoiceSampler(ids, sampler_length)

        # Compiler Transforms:
        transforms = self._transformation_finalizer_by_method(method, transforms, n_channels)

        return self._loader_cls(FullPartTorchDataset(self, transforms, method), batch_size=batch_size,
                                sampler=data_sampler, num_workers=n_workers, pin_memory=pin_memory,
                                collate_fn=completion_collate, drop_last=True)

    def _datapoint_via_full(self, csi):
        return self._full_dict_by_hi(self._hit.csi2chi(csi))

    def _full_dict_by_hi(self, hi):
        v = self._full_path2data(self._hi2full_path(hi))
        if isinstance(v, tuple):
            v, f = v
        else:
            f = self._f
        v = v.astype(self._def_precision)
        return {'gt_hi': hi, 'gt': v, 'f': f}

    def _mask_by_hi(self, hi):
        mask = self._proj_path2data(self._hi2proj_path(hi))
        if len(mask) < self._mask_thresh:
            warn(f'Found mask of length {len(mask)} with id: {hi}')
        return mask

    def _datapoint_via_part(self, si):
        hi = self._hit.si2hi(si)
        d = self._full_dict_by_hi(hi)
        d['gt_mask'] = self._mask_by_hi(hi)
        return d

    # @time_me
    def _build_tupled_index(self):

        # TODO - Revise index map for datasets that are not in-memory
        tup_index_map = []
        for i in range(self.num_projections()):
            for j in range(self.num_full_shapes()):
                if self._hit.csi2chi(j)[0] == self._hit.si2hi(i)[0]:  # Same subject
                    tup_index_map.append((i, j))
        self._tup_index_map = tup_index_map
        # print(convert_bytes(sys.getsizeof(tup_index_map)))

    def _tupled_index_map(self, si):
        return self._tup_index_map[si]

    def _datapoint_via_f2p(self, si):
        si_gt, si_tp = self._tupled_index_map(si)
        tp_dict = self._datapoint_via_full(si_tp)
        gt_dict = self._datapoint_via_part(si_gt)
        gt_dict['tp'], gt_dict['tp_hi'] = tp_dict['gt'], tp_dict['gt_hi']
        return gt_dict

    def _datapoint_via_frand_f2p(self, _):
        # gt_dict = self._datapoint_via_part(si)  # si is gt_si
        gt_hi = self._hit.random_path_from_partial_path()
        gt_dict = self._full_dict_by_hi(gt_hi)
        gt_dict['gt_mask'] = self._mask_by_hi(gt_hi)
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'] = tp_dict['gt'], tp_dict['gt_hi']
        return gt_dict

    def _datapoint_via_rand_f2p(self, si):
        gt_dict = self._datapoint_via_part(si)  # si is gt_si
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'] = tp_dict['gt'], tp_dict['gt_hi']
        return gt_dict

    def _datapoint_via_rand_f2p_seq(self, si):
        gt_dict = self._datapoint_via_part(si)  # si is gt_si
        tp_hi = self._hit.random_path_from_partial_path(gt_dict['gt_hi'][:2])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp'], gt_dict['tp_hi'] = tp_dict['gt'], tp_dict['gt_hi']
        return gt_dict

    def _datapoint_via_rand_ff2p(self, si):
        gt_dict = self._datapoint_via_part(si)  # si is gt_si
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp1'], gt_dict['tp1_hi'] = tp_dict['gt'], tp_dict['gt_hi']

        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])[:-1]  # Shorten hi by 1
        tp_dict = self._full_dict_by_hi(tp_hi)
        gt_dict['tp2'], gt_dict['tp2_hi'] = tp_dict['gt'], tp_dict['gt_hi']

        return gt_dict

    def _datapoint_via_rand_ff2pp(self, si):
        ff2p_dict = self._datapoint_via_rand_ff2p(si)
        # Change gt_mask -> gt_mask1, gt_hi->gt_hi1
        ff2p_dict['gt_mask1'] = ff2p_dict['gt_mask']
        ff2p_dict['gt_hi1'] = ff2p_dict['gt_hi']
        del ff2p_dict['gt_mask'], ff2p_dict['gt_hi']
        # Add in another mask:
        gt_hi2 = self._hit.random_path_from_partial_path(ff2p_dict['gt_hi1'][:-1])  # All but proj id
        ff2p_dict['gt_mask2'] = self._mask_by_hi(gt_hi2)
        ff2p_dict['gt_hi2'] = gt_hi2
        return ff2p_dict

    def _datapoint_via_p2p(self, si):
        si_gt, si_tp = self._tupled_index_map(si)
        tp_dict = self._datapoint_via_part(si_tp)
        gt_dict = self._datapoint_via_part(si_gt)

        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_mask'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_mask']
        return gt_dict

    def _datapoint_via_rand_p2p(self, si):
        gt_dict = self._datapoint_via_part(si)  # si is the gt_si
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])
        tp_dict = self._full_dict_by_hi(tp_hi)
        tp_dict['gt_mask'] = self._mask_by_hi(tp_hi)

        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_mask'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_mask']
        return gt_dict

    def _datapoint_via_frand_p2p(self, _):
        # gt_dict = self._datapoint_via_part(si)  # si is the gt_si
        gt_hi = self._hit.random_path_from_partial_path()
        gt_dict = self._full_dict_by_hi(gt_hi)
        gt_dict['gt_mask'] = self._mask_by_hi(gt_hi)
        tp_hi = self._hit.random_path_from_partial_path([gt_dict['gt_hi'][0]])
        tp_dict = self._full_dict_by_hi(tp_hi)
        tp_dict['gt_mask'] = self._mask_by_hi(tp_hi)

        gt_dict['tp'], gt_dict['tp_hi'], gt_dict['tp_mask'] = tp_dict['gt'], tp_dict['gt_hi'], tp_dict['gt_mask']
        return gt_dict

    def _transformation_finalizer_by_method(self, method, transforms, n_channels):
        if transforms is None:
            transforms = []
        if not isinstance(transforms, list):
            transforms = [transforms]

        if method == 'full':
            align_keys, compiler_keys = ['gt'], None
        elif method == 'part':
            align_keys, compiler_keys = ['gt'], [['gt_part', 'gt_mask', 'gt']]
        elif method in ['f2p', 'rand_f2p', 'frand_f2p', 'rand_f2p_seq']:
            align_keys, compiler_keys = ['gt', 'tp'], [['gt_part', 'gt_mask', 'gt']]
        elif method in ['p2p', 'rand_p2p', 'frand_p2p']:
            align_keys, compiler_keys = ['gt', 'tp'], [['gt_part', 'gt_mask', 'gt'], ['tp_part', 'tp_mask', 'tp']]
        elif method == 'rand_ff2p':
            align_keys, compiler_keys = ['gt', 'tp1', 'tp2'], [['gt_part', 'gt_mask', 'gt']]
        elif method == 'rand_ff2pp':
            align_keys, compiler_keys = ['gt', 'tp1', 'tp2'], \
                                        [['gt_part1', 'gt_mask1', 'gt'], ['gt_part2', 'gt_mask2', 'gt']]
        else:
            raise AssertionError

        transforms.insert(0, AlignChannels(keys=align_keys, n_channels=n_channels, uni_faces=self._f is not None))
        if compiler_keys is not None:
            transforms.append(PartCompiler(compiler_keys))

        return Compose(transforms)

    def _hi2proj_path(self, hi):
        raise NotImplementedError

    def _hi2full_path(self, hi):
        raise NotImplementedError

    def _proj_path2data(self, fp):
        raise NotImplementedError

    def _full_path2data(self, fp):
        raise NotImplementedError

    # TODO - Use this to rewrite show_sample
    # def show_sample(self, n_shapes=8, key='gt_part', strategy='spheres', with_vnormals=False, *args, **kwargs):
    #     from mesh.plot import plot_mesh_montage
    #     assert strategy in ['spheres', 'mesh', 'cloud']
    #     using_full = key in ['gt', 'tp']
    #
    #     fp_fun = self._hi2full_path if using_full else self._hi2proj_path
    #     samp = self.sample(n_shapes)
    #
    #     origin = key.split('_')[0]
    #     labelb = [f'{key} | {fp_fun(samp[f"{origin}_hi"][i]).name}' for i in range(n_shapes)]
    #     vb = samp[key][:, :, 0:3].numpy()
    #     if with_vnormals:  # TODO - add intergration for gt_part
    #         nb = samp[key][:, :, 3:6].numpy()
    #     else:
    #         nb = None
    #
    #     if strategy == 'mesh':
    #         if using_full:
    #             fb = self._f
    #         else:
    #             # TODO - Should we change the vertices as well?
    #             fb = [trunc_to_vertex_mask(vb[i], self._f, samp[f'{origin}_mask_vi'][i])[1] for i in range(n_shapes)]
    #     else:
    #         fb = None
    #
    #     plot_mesh_montage(vb=vb, fb=fb, nb=nb, labelb=labelb, spheres_on=(strategy == 'spheres'),
    #                       *args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class FullPartTorchDataset(Dataset):
    SAME_SHAPE_RETRY_BOUND = 3
    RETRY_BOUND = SAME_SHAPE_RETRY_BOUND * 7  # Try to retry atleast 7 shapes before dying

    # Note that changes to Dataset will be seen in any loader derived from it before
    # This should be taken into account when decimating the Dataset index
    def __init__(self, ds_inst, transforms, method):
        self._ds_inst = ds_inst
        self._transforms = transforms
        self._method = method
        self.get_func = getattr(self._ds_inst, f'_datapoint_via_{method}')
        self.self_len = self._ds_inst.num_datapoints_by_method(self._method)
        self.use_unsafe_meth = not self._ds_inst._suspected_corrupt

    def __len__(self):
        return self.self_len

    def __getitem__(self, si):
        if self.use_unsafe_meth:
            return self._transforms(self.get_func(si))
        else:  # This is a hack to enable reloading
            global_retries = 0
            local_retries = 0
            while 1:
                try:
                    return self._transforms(self.get_func(si))
                except Exception as e:
                    global_retries += 1
                    local_retries += 1
                    if global_retries == self.RETRY_BOUND:
                        raise e
                    if local_retries == self.SAME_SHAPE_RETRY_BOUND:
                        local_retries = 0
                        si += 1  # TODO - Find better way
                        if si == self.self_len:  # Double back
                            si = 0


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class ParametricCompletionDataset(FullPartCompletionDataset, ABC):
    # This adds the assumption that each mesh has the same connectivity, and the same number of vertices
    def __init__(self, n_verts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self._data_dir / 'face_template.pkl', "rb") as f_file:
            self._f = load(f_file)
            self._f.flags.writeable = False  # Make this a read-only numpy array

        self._n_v = n_verts
        self._null_shape = None
        self._loader_cls = ParametricLoader

    def faces(self):
        return self._f

    def num_verts(self):
        return self._n_v

    def num_faces(self):
        return self._f.shape[0]

    def null_shape(self, n_channels):
        # Cache shape:
        if self._null_shape is None or self._null_shape.shape[1] != n_channels:
            self._null_shape = align_channels(self._datapoint_via_full(0)['gt'], self._f, n_channels)
            self._null_shape.flags.writeable = False

        return self._null_shape

    def plot_null_shape(self, strategy='mesh', with_vnormals=False):
        null_shape = self.null_shape(n_channels=6)
        n = null_shape[:, 3:6] if with_vnormals else None
        plot_mesh(v=null_shape[:, :3], f=self._f, n=n, strategy=strategy)


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyUnresolvedReferences
def completion_collate(batch, stop=False):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    if stop:
        return batch
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(
                    f"default_collate: batch must contain tensors, numpy arrays, "
                    f"numbers, dicts or lists; found {elem.dtype}")

            return completion_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        # A bit hacky - but works
        d = {}
        for k in elem:
            for suffix in ['_hi', '_mask', '_mask1', '_mask2']:  # TODO
                if k.endswith(suffix):
                    stop = True
                    break
            else:
                stop = False
            d[k] = completion_collate([d[k] for d in batch], stop)
        return d

        # return {key: default_collate([d[key] for d in batch],rec_level=1) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(completion_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [completion_collate(samples) for samples in transposed]
    raise TypeError(
        f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {elem.dtype}")


np_str_obj_array_pattern = re.compile(r'[SaUO]')
