from collections import OrderedDict
from collections.abc import MutableMapping
from bisect import bisect_right
from util.container import max_dict_depth, min_dict_depth, deep_dict_to_rdict
from util.string_op import banner, warn
from json import dumps
import random
from copy import deepcopy


# from sortedcontainers import SortedDict

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Index Tree Mapping
# ----------------------------------------------------------------------------------------------------------------------
class HierarchicalIndexTree:
    def __init__(self, hit, in_memory):

        # Seems that dict() maintains insertion order by default from Py 3.7>,
        # and I'm presuming Python 3.7 is used to run this code
        # if not deep_check_odict(hit):
        # warn("Hierarchical Index Tree provided is not ordered - using sorted() order",stacklevel=2)
        # hit = deep_dict_to_rdict(hit)
        if not hit:
            warn("Hierarchical Index Tree is empty", stacklevel=2)
        max_depth = max_dict_depth(hit)
        min_depth = min_dict_depth(hit)
        assert max_depth == min_depth, "Hierarchical Index Tree must be balanced"

        self._rhit = deep_dict_to_rdict(hit)  # Not very efficient to duplicate the tree, but meh
        self._hit = hit
        self._in_memory = in_memory
        self._depth = max_depth
        # multid has two possible forms, in dependence of in_memory
        self._multid_list, self._num_objs = self._flatten(self._in_memory)
        self._csi_list = None

    def __str__(self):
        # pf = pformat(self._hit)
        # return pf.replace('OrderedDict', 'HierarchicalIndexTree')
        # TODO - While this returns a nice string - it changes the tree into the json format, which
        # does not have ints as keys
        return 'HierarchicalIndexTree' + dumps(self._hit, indent=2)

    def depth(self):
        return self._depth

    def num_indexed(self):
        return self._num_objs

    def num_index_clusters(self):
        if self._csi_list is None:
            self.init_cluster_hi_list()
        return len(self._csi_list)

    def random_path_from_partial_path(self, partial_path=tuple()):
        # Init
        partial_tree = self._rhit
        path = list(partial_path)

        # Advance through the tree exactly len(partial_path) ids
        for index in partial_path:
            partial_tree = partial_tree[index]
        # Advance to leaves at random
        while isinstance(partial_tree, MutableMapping):
            rkey = partial_tree.random_key()
            path.append(rkey)
            partial_tree = partial_tree[rkey]
        # Handle last value: [Remember, range is inclusive]
        path.append(random.randint(0, partial_tree - 1))
        return tuple(path)

    def get_id_union_by_depth(self, depth):
        if depth > self._depth - 1 or depth <= 0:
            # We don't allow getting of ids from the lowest level - seeing they are not unique
            raise AssertionError(f"Invalid depth requested. Valid inputs are: {list(range(1, self._depth))}")

        key_set = set()
        self._get_id_union_by_depth(self._hit, depth - 1, key_set)
        return sorted(list(key_set))

    def keep_ids_by_depth(self, keepers, depth):
        raise NotImplementedError  # TODO - Implement this - along with dataset decimation

    def remove_ids_by_depth(self, goners, depth):
        if depth > self._depth - 1 or depth <= 0:
            # We don't allow getting of ids from the lowest level - seeing they are not unique
            raise AssertionError(f"Invalid depth requested. Valid inputs are: {list(range(1, self._depth))}")

        if type(goners) is not list:
            goners = [goners]
        reduced_hit = self._remove_ids_by_depth(self._hit, depth - 1, set(goners))
        return HierarchicalIndexTree(reduced_hit, self._in_memory)

    def init_cluster_hi_list(self):
        self._csi_list = self._flatten(in_memory=False)[0][1]

    def in_memory(self):
        return self._in_memory

    def csi2chi(self, csi):
        return self._csi_list[csi]

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

    def _flatten(self, in_memory):

        flat_hit = self._flatten_aux(self._hit, in_memory)
        flat_hit = [tuple(o) for o in flat_hit]  # Turn into tuple for correctness
        flat_hit = tuple(flat_hit)

        if in_memory:
            return flat_hit, len(flat_hit)
        else:
            acc_list = [[], []]  # Accumulation list, multi-index per bin
            tot_obj_cnt = 0
            for multindex in flat_hit:
                acc_list[0].append(tot_obj_cnt)
                acc_list[1].append(multindex[0:len(multindex) - 1])
                tot_obj_cnt += multindex[-1]
                # Increase the accumulation by the last member in multilevel index tree - the object cnt
            return acc_list, tot_obj_cnt

    def init_tuple_search_tree(self):
        # TODO - Finish this
        search_tree = self._build_tuple_search_tree_aux(deepcopy(self._hit), 0)

    @staticmethod
    def _build_tuple_search_tree_aux(hit, ind=0):
        # TODO - Finish this
        if isinstance(hit, MutableMapping):
            local_list = []
            i = ind
            for key, value in hit.items():
                if not isinstance(value, MutableMapping):
                    hit.pop(key)  # Destroy the leaves
                else:
                    i = HierarchicalIndexTree._build_tuple_search_tree_aux(value, i)
                    local_list.append(i)
        else:  # Value
            return ind

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


# ----------------------------------------------------------------------------------------------------------------------
#                                    Utility Methods - TODO - migrate to Data Prep
# ----------------------------------------------------------------------------------------------------------------------

def construct_dfaust_hit_pickle(with_write=False):
    import re
    from cfg import PRIMARY_DATA_DIR
    from pathlib import Path
    dataset_fp = Path(PRIMARY_DATA_DIR) / 'synthetic' / 'DFaustPyProj'
    fp = dataset_fp / "dfaust_subjects_and_sequences.txt"
    assert fp.is_file(), "Could not find Dynamic Faust subjects_and_sequences.txt file"
    print(f'Found dfaust subjects and sequence list at {fp.parents[0].resolve()}')
    with open(fp) as f:
        lines = [line.rstrip('\n') for line in f]

    hit = {}
    last_subj = None

    for line in lines:
        m = re.match(r"(\d+)\s+\((\w+)\)", line)
        if m:  # New hit
            sub_id, _ = m.group(1, 2)
            last_subj = sub_id
            hit[last_subj] = {}
        elif line.strip():
            seq, frame_cnt = line.split()
            frame_cnt = int(frame_cnt)
            hit[last_subj][seq] = {}
            for i in range(frame_cnt):
                hit[last_subj][seq][i] = 10

    hit = HierarchicalIndexTree(hit, in_memory=False)
    if with_write:
        from pickle import dump
        pkl_fp = dataset_fp / "DFaust_hit.pkl"
        with open(pkl_fp, "wb") as f:
            dump(hit, f)
        print(f'Dumped Dynamic Faust HIT Pickle at {pkl_fp.resolve()}')
    return hit


def construct_amass_hit_pickles(with_write=False):
    from pathlib import Path
    import json
    from cfg import PRIMARY_DATA_DIR
    train_fp = Path(PRIMARY_DATA_DIR) / 'synthetic' / 'AmassTrainPyProj' / 'train_dict.json'
    vald_fp = Path(PRIMARY_DATA_DIR) / 'synthetic' / 'AmassValdPyProj' / 'vald_dict.json'
    test_fp = Path(PRIMARY_DATA_DIR) / 'synthetic' / 'AmassTestPyProj' / 'test_dict.json'
    hits = []
    for fp, appender in zip([train_fp, vald_fp, test_fp], ['train', 'vald', 'test']):
        with open(fp.resolve()) as f:
            hit = json.loads(f.read())

        # Transform:
        ks = list(hit.keys())
        for k in ks:
            val = hit.pop(k)
            k = int(k)
            hit[k] = dict()
            for i in range(val):
                hit[k][i] = 10

        if fp == test_fp:
            hit = HierarchicalIndexTree(hit, in_memory=True)
        else:
            hit = HierarchicalIndexTree(hit, in_memory=False)
        hits.append(hit)

        if with_write:
            from pickle import dump
            tgt_fp = fp.parents[0] / f'amass_{appender}_hit.pkl'
            with open(tgt_fp, "wb") as file:
                dump(hit, file)
                print(f'Created index at {tgt_fp.resolve()}')
    return hits


def generic_fs_hit_build(fp):
    from pathlib import Path
    from pprint import pprint
    fp = Path(fp)
    assert fp.is_dir(), f"{fp} is not a directory"
    hit = {}
    _generic_fs_hit_build_aux(fp, hit)
    pprint(hit)


def _generic_fs_hit_build_aux(fp, hit):
    for x in fp.iterdir():
        if x.is_file():
            num_files = len([name for name in fp.iterdir() if name.is_file()])
            return num_files
        else:
            hit[x.name] = _generic_fs_hit_build_aux(fp / x.name, {})

    return hit


# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Modules
# ----------------------------------------------------------------------------------------------------------------------

def hit_test():
    # import time
    from pprint import pprint
    # print = (lambda p: lambda *args, **kwargs: [p(*args, **kwargs), time.sleep(.01)])(print)
    hit = {
        'Subject1': {
            'Pose1': 200,
            'Pose2': 300,
            # 'Pose3': {
            #     'Seq1': 500,
            #     'Seq2': 600
            # }
        },
        'Subject2': {
            'Pose1': 1,
            'Pose2': 4,
            'Pose3': 4,
        },
        'Subject3': {
            'Pose1': 100,
            'Pose2': 50,
        },
        # 'Subject 4': 3
    }

    hit_mem = HierarchicalIndexTree(hit, in_memory=True)
    hit_out_mem = HierarchicalIndexTree(hit, in_memory=False)

    banner('In Memory vs Out of Memory Tests')
    print(hit_mem)
    print(hit_mem.depth())
    print(hit_mem.num_indexed())
    print(hit_out_mem)
    print(hit_out_mem.depth())
    print(hit_out_mem.num_indexed())

    ids = range(hit_mem.num_indexed())
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
    banner('Random Tests')
    print(hit_mem.random_path_from_partial_path())
    print(hit_mem.random_path_from_partial_path())
    print(hit_mem.random_path_from_partial_path(('Subject1', 'Pose2')))


def hit_test2():
    def construct_faust_hit():
        hit = {}
        for sub_id in range(10):
            hit[sub_id] = {}
            for pose_id in range(10):
                hit[sub_id][pose_id] = 10
        return HierarchicalIndexTree(hit, in_memory=True)

    hit = construct_dfaust_hit_pickle()
    hit.init_cluster_hi_list()
    for i in range(hit.num_index_clusters()):
        print(hit.csi2chi(i))
    print(hit)

    # hit_test()
    # example_hit = construct_dfaust_hit_pickle()
    # print(example_hit)
    # print(example_hit.num_objects())
    # construct_amass_hit_pickles()


def generic_hit():
    fp = r'C:\Users\idoim\Desktop\ShapeCompletion\data\synthetic\DFaustPyProj\full'
    generic_fs_hit_build(fp)


if __name__ == "__main__":
    generic_hit()
