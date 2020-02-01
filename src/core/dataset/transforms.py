import random
from mesh.ops import vnrmls, moments, padded_part_by_mask, flip_vertex_mask
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
#                                                  Transforms- Abstract
# ----------------------------------------------------------------------------------------------------------------------

class Transform:
    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def append(self, new_transform):
        self.transforms.append(new_transform)

    def insert(self, index, new_transform):
        self.transforms.insert(index, new_transform)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# ----------------------------------------------------------------------------------------------------------------------
#                                               Special Transforms
# ----------------------------------------------------------------------------------------------------------------------
class AlignInputChannels(Transform):
    def __init__(self, req_in_channels):
        self._req_in_channels = req_in_channels

    def __call__(self, x):
        x['gt'] = align_in_channels(x['gt'], x['f'], self._req_in_channels)
        x['tp'] = align_in_channels(x['tp'], x['f'], self._req_in_channels)
        # if self._req_in_channels < 6:
        del x['f']  # Remove this as an optimization
        return x


class PartCompiler(Transform):
    def __init__(self, part_keys):
        self._part_keys = part_keys

    def __call__(self, x):
        # Done last, since we might transform the mask
        for (k_part, k_mask, k_full) in self._part_keys:
            x[k_part] = padded_part_by_mask(x[k_mask], x[k_full])
        return x


# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Augmentation Transforms
# ----------------------------------------------------------------------------------------------------------------------

class RandomMaskFlip(Transform):
    def __init__(self, prob):  # Probability of mask flip
        self._prob = prob

    def __call__(self, x):
        if random.random() < self._prob:
            nv = x['gt'].shape[0]
            x['gt_mask_vi'] = flip_vertex_mask(nv, x['gt_mask_vi'])
            # TODO: tp mask flips?
        return x


class Center(Transform):
    def __init__(self, slicer=slice(0, 3)):
        self._slicer = slicer

    def __call__(self, x):
        x['gt'][:, self._slicer] -= x['gt'][:, self._slicer].mean(axis=0, keepdims=True)
        x['tp'][:, self._slicer] -= x['tp'][:, self._slicer].mean(axis=0, keepdims=True)
        return x


class UniformVertexScale(Transform):
    def __init__(self, scale):
        self._scale = scale

    def __call__(self, x):
        x['gt'][:, 0:3] *= self._scale
        x['tp'][:, 0:3] *= self._scale
        return x


# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Augmentation Transforms
# ----------------------------------------------------------------------------------------------------------------------

def align_in_channels(v, f, req_in_channels):
    available_in_channels = v.shape[1]
    if available_in_channels > req_in_channels:
        return v[:, 0:req_in_channels]
    else:
        combined = [v]
        if req_in_channels >= 6 > available_in_channels:
            combined.append(vnrmls(v, f))
        if req_in_channels >= 12 > available_in_channels:
            combined.append(moments(v))

        return np.concatenate(combined, axis=1)
