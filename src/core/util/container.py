from collections.abc import MutableMapping
from collections import OrderedDict
import random
# ----------------------------------------------------------------------------------------------------------------------
#                                                Dictionary
# ----------------------------------------------------------------------------------------------------------------------
def max_dict_depth(dic, level=1):
    if not isinstance(dic, dict) or not dic:
        return level
    return max(max_dict_depth(dic[key], level + 1)
               for key in dic)

def min_dict_depth(dic, level=1):
    if not isinstance(dic, dict) or not dic:
        return level
    return min(min_dict_depth(dic[key], level + 1)
               for key in dic)

def deep_dict_to_rdict(d):
    rdict = RandomDict()

    for key in sorted(d.keys()):
        value = d[key]
        if isinstance(value, dict):
            rdict[key] = deep_dict_to_rdict(value)
        else:
            rdict[key] = value

    return rdict

def deep_check_odict(d):

    if not isinstance(d,OrderedDict):
        return False
    else:
        kids_are_ordered = True
        for v in d.values():
            if isinstance(v, MutableMapping):
                kids_are_ordered &= deep_check_odict(v)
        return kids_are_ordered


def delete_keys_from_dict(dictionary, keys):
    keys_set = set(keys)  # Just an optimization for the "if key in keys" lookup.

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            else:
                modified_dict[key] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
    return modified_dict

# ----------------------------------------------------------------------------------------------------------------------
#                                                Dictionary
# ----------------------------------------------------------------------------------------------------------------------
class RandomDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        """ Create RandomDict object with contents specified by arguments.
        Any argument
        :param *args:       dictionaries whose contents get added to this dict
        :param **kwargs:    key, value pairs will be added to this dict
        """
        # mapping of keys to array positions
        self.keys = {}
        self.values = []
        self.last_index = -1

        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        if key in self.keys:
            i = self.keys[key]
        else:
            self.last_index += 1
            i = self.last_index

        self.values.append((key, val))
        self.keys[key] = i

    def __delitem__(self, key):
        if not key in self.keys:
            raise KeyError

        # index of item to delete is i
        i = self.keys[key]
        # last item in values array is
        move_key, move_val = self.values.pop()

        if i != self.last_index:
            # we move the last item into its location
            self.values[i] = (move_key, move_val)
            self.keys[move_key] = i
        # else it was the last item and we just throw
        # it away

        # shorten array of values
        self.last_index -= 1
        # remove deleted key
        del self.keys[key]

    def __getitem__(self, key):
        if not key in self.keys:
            raise KeyError

        i = self.keys[key]
        return self.values[i][1]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return self.last_index + 1

    def random_key(self):
        """ Return a random key from this dictionary in O(1) time """
        if len(self) == 0:
            raise KeyError("RandomDict is empty")

        i = random.randint(0, self.last_index)
        return self.values[i][0]

    def random_value(self):
        """ Return a random value from this dictionary in O(1) time """
        return self[self.random_key()]

    def random_item(self):
        """ Return a random key-value pair from this dictionary in O(1) time """
        k = self.random_key()
        return k, self[k]