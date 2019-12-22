from collections.abc import MutableMapping
from collections import OrderedDict
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

def deep_dict_to_odict(d):
    ordered_dict = OrderedDict()

    for key in sorted(d.keys()):
        value = d[key]
        if isinstance(value, dict):
            ordered_dict[key] = deep_dict_to_odict(value)
        else:
            ordered_dict[key] = value

    return ordered_dict

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