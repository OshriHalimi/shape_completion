import torch
import numpy as np
import torch.nn.functional as tf
from util.datascience import normr, index_sparse
from util.gen import warn


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class CompletionTriplet:
    def __init__(self, f, hi, gt_v, mask_vi, new_hi, tp_v, tp_mask_vi=None):
        self.hi = hi
        self.gt_v = gt_v
        self.mask_vi = mask_vi
        self.new_hi = new_hi
        self.tp_v = tp_v

        self.tp_mask_vi = tp_mask_vi  # Sometimes None
        self.f = f

        # Placeholders:
        self.mask_penalty_vec = None


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class Transform:
    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

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
#
# ----------------------------------------------------------------------------------------------------------------------
class AlignInputChannels(Transform):
    def __init__(self, req_in_channels):
        self._req_in_channels = req_in_channels

    def __call__(self, data):
        if isinstance(data, CompletionTriplet):  # Assumption: tp_v,gt_v use the same triangulation
            data.gt_v = align_in_channels(data.gt_v, data.f, self._req_in_channels)
            data.tp_v = align_in_channels(data.tp_v, data.f, self._req_in_channels)
        else:
            raise NotImplementedError

        return data


class CompletionTripletToTuple(Transform):
    def __init__(self):
        # To avoid circular import errors:
        from cfg import DANGEROUS_MASK_THRESH, DEF_PRECISION
        self._mask_thresh = DANGEROUS_MASK_THRESH
        self._def_prec = DEF_PRECISION

    def __call__(self, data):
        if isinstance(data, CompletionTriplet):
            # Checks:
            if len(data.mask_vi) < self._mask_thresh:
                warn(f'Found mask of length {len(data.mask_vi)} with id: {data.hi}')
            if data.tp_mask_vi is not None and len(data.mask_vi) < self._mask_thresh:
                warn(f'Found mask of length {len(data.tp_mask_vi)} with id: {data.new_hi}')

            # ONLY PLACE THAT PRECISION IS CHANGED
            data.gt_v = data.gt_v.astype(self._def_prec)
            data.tp_v = data.tp_v.astype(self._def_prec)

            output = [data.hi, data.gt_v, padded_part_by_mask(data.mask_vi, data.gt_v), data.new_hi, data.tp_v]

            if data.tp_mask_vi is not None:
                output.append(padded_part_by_mask(data.tp_mask_vi, data.tp_v))

            # Additional input - outside of the input config
            if data.mask_penalty_vec is not None:
                output.append(data.mask_penalty_vec)

            return tuple(output)
        else:
            raise NotImplementedError


class Center(Transform):
    def __init__(self, slicer=slice(0, 3)):
        self._slicer = slicer

    def __call__(self, data):
        if isinstance(data, CompletionTriplet):
            data.gt_v[:, self._slicer] -= data.gt_v[:, self._slicer].mean(axis=0, keepdims=True)
            data.tp_v[:, self._slicer] -= data.tp_v[:, self._slicer].mean(axis=0, keepdims=True)
        else:
            raise NotImplementedError
        return data


class AddMaskPenalty(Transform):
    def __init__(self, penalty):
        self._penalty = penalty

    def __call__(self, data):
        if isinstance(data, CompletionTriplet):
            data.mask_penalty_vec = np.ones(data.gt_v.shape[0])
            data.mask_penalty_vec[data.mask_vi] = self._penalty
        else:
            raise NotImplementedError
        return data


# ----------------------------------------------------------------------------------------------------------------------#
#                                           Transform Inner Functions
# ----------------------------------------------------------------------------------------------------------------------#

def padded_part_by_mask(mask_vi, gt_v):
    # Pad the mask to length:
    needed_padding_len = gt_v.shape[0] - len(mask_vi)
    mask_vi_padded = np.append(mask_vi, np.random.choice(mask_vi, needed_padding_len, replace=True))  # Copies
    return gt_v[mask_vi_padded, :]


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
#     def __call__(self, data):
#         assert 'face' in data
#         pos, face = data.pos, data.face
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
#         data.norm = norm
#
#         return data
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
