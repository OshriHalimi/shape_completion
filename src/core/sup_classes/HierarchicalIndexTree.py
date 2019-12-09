import collections
from bisect import bisect_right
from pprint import pprint, pformat
from utils.container_utils import dict_depth
from utils.gen_utils import banner
import warnings


# ----------------------------------------------------------------------------------------------------------------------
#                                                  Index Tree Mapping
# ----------------------------------------------------------------------------------------------------------------------
class HierarchicalIndexTree:
    def __init__(self, hit, in_memory):
        if not isinstance(hit, collections.OrderedDict):
            warnings.warn("Hierarchical Index Tree provided is not ordered")
            hit = collections.OrderedDict(hit)  # Better late than never
        self._hit = hit
        self._in_memory = in_memory
        self._depth = dict_depth(self._hit)
        self._multid_list, self._num_objs = self._flatten()  # multid has two possible forms, in dependence of in_memory

    def __str__(self):
        pf = pformat(self._hit)
        return pf.replace('OrderedDict', 'HierarchicalIndexTree')

    def depth(self):
        return self._depth

    def num_objects(self):
        return self._num_objs

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
    def _flatten_aux(hit, extended_flatten):
        if isinstance(hit, collections.abc.MutableMapping):
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
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------

def test_module():
    hit = {'Subject 1':
               {'Pose1': 200,
                'Pose2': 300,
                'Pose3': {
                    'Seq1': 500,
                    'Seq2': 600
                }
                },
           'Subject2':
               {'Pose1': 1,
                'Pose2': 4,
                'Pose3': 4,
                },
           'Subject3':
               {'Pose1': 100,
                'Pose2': 50,
                }
           }

    hit_mem = HierarchicalIndexTree(hit, in_memory=True)
    hit_out_mem = HierarchicalIndexTree(hit, in_memory=False)

    print(hit_mem)
    print(hit_mem.depth())
    print(hit_mem.num_objects())
    print(hit_out_mem)
    print(hit_out_mem.depth())
    print(hit_out_mem.num_objects())

    ids = range(hit_mem.num_objects())
    for id in ids:
        # print(hit_mem.si2hi(id))
        # print(hit_out_mem.si2hi(id))
        assert(hit_mem.si2hi(id) == hit_out_mem.si2hi(id))
    else:
        banner('Sequence ID Test Passed')

# ----------------------------------------------------------------------------------------------------------------------
#                                                Test Module
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__": test_module()
