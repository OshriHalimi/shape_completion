import torch
import numpy as np
import torch.nn.functional as tf
from util.datascience import normr, index_sparse
from util.gen import warn
import random


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
        x['gt_v'] = align_in_channels(x['gt_v'], x['f'], self._req_in_channels)
        x['tp_v'] = align_in_channels(x['tp_v'], x['f'], self._req_in_channels)
        del x['f']  # TODO - Change this - Presumption: The triangulation is not needed after this
        return x


class PartCompiler(Transform):
    def __init__(self, part_keys):
        self._part_keys = part_keys

    def __call__(self, x):
        # Done last, since we might transform the mask
        for (k_part, k_mask, k_full) in self._part_keys:
            x[k_part] = padded_part_by_mask(x[k_mask][0], x[k_full])
        return x


# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Augmentation Transforms
# ----------------------------------------------------------------------------------------------------------------------

class RandomMaskFlip(Transform):
    def __init__(self, prob):  # Probability of mask flip
        self._prob = prob

    def __call__(self, x):
        if random.random() < self._prob:
            nv = x['gt_v'].shape[0]
            x['gt_mask_vi'] = flip_mask(nv, x['gt_mask_vi'])
            # TODO: tp mask flips?
        return x


class Center(Transform):
    def __init__(self, slicer=slice(0, 3)):
        self._slicer = slicer

    def __call__(self, x):
        x['gt_v'][:, self._slicer] -= x['gt_v'][:, self._slicer].mean(axis=0, keepdims=True)
        x['tp_v'][:, self._slicer] -= x['tp_v'][:, self._slicer].mean(axis=0, keepdims=True)
        return x


class UniformVertexScale(Transform):
    def __init__(self, scale):
        self._scale = scale

    def __call__(self, x):
        x['gt_v'][:, 0:3] *= self._scale
        x['tp_v'][:, 0:3] *= self._scale
        return x


# ----------------------------------------------------------------------------------------------------------------------#
#                                           Transform Inner Functions
# ----------------------------------------------------------------------------------------------------------------------#

def padded_part_by_mask(mask_vi, v):
    # Pad the mask to length:
    needed_padding_len = v.shape[0] - len(mask_vi)
    mask_vi_padded = np.append(mask_vi, np.random.choice(mask_vi, needed_padding_len, replace=True))  # Copies
    return v[mask_vi_padded, :]


def align_in_channels(v, f, req_in_channels):
    available_in_channels = v.shape[1]
    if available_in_channels > req_in_channels:
        return v[:, 0:req_in_channels]
    else:
        combined = [v]
        if req_in_channels >= 6 > available_in_channels:
            combined.append(calc_vnrmls(v, f))
        if req_in_channels >= 12 > available_in_channels:
            combined.append(calc_moments(v))

        return np.concatenate(combined, axis=1)


def calc_vnrmls(v, f):
    # NOTE - Vertices unreferenced by faces will be zero
    if f is None:
        raise NotImplementedError  # TODO - Add in computation for scans, without faces - either with pcnormals/
    else:
        a = v[f[:, 0], :]
        b = v[f[:, 1], :]
        c = v[f[:, 2], :]
        fn = np.cross(b - a, c - a)
        matrix = index_sparse(v.shape[0], f)
        vn = matrix.dot(fn)
        return normr(vn)


def calc_moments(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)


def flip_mask(nv, vi):
    indicator = np.ones((nv,))
    indicator[vi] = 0
    return np.where(indicator == 1)[0]


def trunc_to_vertex_subset(v, f, vi):
    nv = v.shape[0]
    # Compute new vertices:
    v2 = v[vi, :]
    # Compute map from old vertex indices to new vertex indices
    vlut = np.full((nv,), fill_value=-1)
    vlut[vi] = np.arange(len(vi))  # Bad vertices have no mapping, and stay -1.

    # Change vertex labels in face array. Bad vertices have no mapping, and stay -1.
    f2 = vlut[f]
    # Keep only faces with valid vertices:
    f2 = f2[np.sum(f2 == -1, axis=1) == 0, :]
    return v2, f2


# ----------------------------------------------------------------------------------------------------------------------#
#                                               PyTorch Batch Computations - TODO - Migrate this
# ----------------------------------------------------------------------------------------------------------------------#

def batch_euclidean_dist_matrix(x):
    # X of dim: [batch_size x nv x 3]
    x = x.transpose(2, 1)
    r = torch.sum(x ** 2, dim=2).unsqueeze(2)  # [batch_size  x num_points x 1]
    r_t = r.transpose(2, 1)  # [batch_size x 1 x num_points]
    inner = torch.bmm(x, x.transpose(2, 1))
    dist_mat = tf.relu(r - 2 * inner + r_t) ** 0.5  # the residual numerical error can be negative ~1e-16
    return dist_mat


def batch_vnrmls(v, f_tup):
    # v dimensions: [batch_size x 3 x n_vertices]
    # f dimensions: ( [n_faces x 3] , [n_faces x 3] )
    v = v.transpose(2, 1)
    vn = torch.zeros_like(v)
    for i in range(v.shape[0]):
        vn[i, :, :] = calc_vnrmls_torch(v[i, :, :], f_tup)

    v = v.transpose(2, 1)
    vn = vn.transpose(2, 1)
    return vn


def calc_vnrmls_torch(v, f_tup):
    f, f_torch = f_tup

    a = v[f_torch[:, 0], :]
    b = v[f_torch[:, 1], :]
    c = v[f_torch[:, 2], :]
    fn = torch.cross(b - a, c - a)

    matrix = index_sparse(v.shape[0], f)
    matrix = torch.from_numpy(matrix.todense()).float().cuda()
    vn = torch.mm(matrix, fn)
    # Normalize them
    # Note - in some runs I've made, vectors computed are degenrate and cause errors in the computation.
    # The normr function masks these - I.I.
    # vn = vn / np.sqrt(np.sum(vn ** 2, -1, keepdims=True)) # Does not handle 0 vectors
    vn = tf.normalize(vn, p=2, dim=1)
    # vn = normr(vn)
    # Old Vertex Normals
    # vn = np.zeros_like(v)
    # vn[self.ref_tri[:, 0], :] = vn[self.ref_tri[:, 0], :] + fn
    # vn[self.ref_tri[:, 1], :] = vn[self.ref_tri[:, 1], :] + fn
    # vn[self.ref_tri[:, 2], :] = vn[self.ref_tri[:, 2], :] + fn

    return vn


# import torch
# import torch.nn.functional as F
# from torch_scatter import scatter_add
# TODO - See if this implementation is faster for vnrmrls
# [docs]class GenerateMeshNormals(object):
#     r"""Generate normal vectors for each mesh node based on neighboring
#     faces."""
#
#     def __call__(self, x):
#         assert 'face' in x
#         pos, face = x.pos, x.face
#
#         vec1 = pos[face[1]] - pos[face[0]]
#         vec2 = pos[face[2]] - pos[face[0]]
#         face_norm = F.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]
#
#         idx = torch.cat([face[0], face[1], face[2]], dim=0)
#         face_norm = face_norm.repeat(3, 1)
#
#         norm = scatter_add(face_norm, idx, dim=0, dim_size=pos.size(0))
#         norm = F.normalize(norm, p=2, dim=-1)  # [N, 3]
#
#         x.norm = norm
#
#         return x
#
#     def __repr__(self):
#         return '{}()'.format(self.__class__.__name__)


# ----------------------------------------------------------------------------------------------------------------------#
#                                              Tests
# ----------------------------------------------------------------------------------------------------------------------#

def test_normals(v, f, n):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=0.2, antialiased=True)
    vnn = v + n
    ax.quiver(v[:, 0], v[:, 1], v[:, 2], vnn[:, 0], vnn[:, 1], vnn[:, 2], length=0.03, normalize=True)
    plt.show()
