import torch
import numpy as np
import torch.nn.functional as F
from util.datascience import normr, index_sparse
from util.gen import warn

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Globals
# ----------------------------------------------------------------------------------------------------------------------
DEF_PRECISION = np.float32
DANGEROUS_MASK_THRESH = 100

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

class CompletionPair:
    def __init__(self, gt_v, mask_vi,hi, f=None):
        self.gt_v = gt_v
        self.mask_vi = mask_vi
        self.hi = hi
        self.f = f

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
        if isinstance(data, CompletionPair):
            available_in_channels = data.gt_v.shape[1]
            if available_in_channels > self._req_in_channels:
                data.gt_v = data.gt_v[:, 0:self._req_in_channels]
            else:
                final_gt = [data.gt_v]
                if self._req_in_channels >= 6 > available_in_channels:
                    final_gt.append(calc_vnrmls(data.gt_v, data.f))
                if self._req_in_channels >= 12 > available_in_channels:
                    final_gt.append(calc_moments(data.gt_v))

                data.gt_v = np.concatenate(final_gt, axis=1)
        else:
            raise NotImplementedError

        return data


class CompletionPairToTuple(Transform):
    def __call__(self, data):
        if isinstance(data, CompletionPair):
            # Checks:
            if len(data.mask_vi) < DANGEROUS_MASK_THRESH:
                warn(f'Found mask of length {len(data.mask_vi)} with id: {data.hi}')
            data.gt_V = data.gt_v.as_type(DEF_PRECISION) # ONLY PLACE THAT PRECISION IS CHANGED

            return (data.gt_v, padded_part_by_mask(data.mask_vi, data.gt_v),data.mask_penalty_vec)
        else:
            raise NotImplementedError

class Center(Transform):
    def __init__(self,slicer=slice(0,3)):
        self._slicer = slicer

    def __call__(self, data):
        if isinstance(data, CompletionPair):
            data.gt_v[:,self._slicer] -= data.gt_v[:,self._slicer].mean(axis=0,keepdims=True)
        else:
            raise NotImplementedError
        return data

class AddMaskPenalty(Transform):
    def __init__(self,penalty):
        self._penalty = penalty

    def __call__(self, data):
        if isinstance(data, CompletionPair):
            data.mask_penalty_vec = np.ones(data.gt_v.shape[0])
            data.mask_penalty_vec[data.mask_vi] = self.penalty
        else:
            raise NotImplementedError
        return data


# ----------------------------------------------------------------------------------------------------------------------#
#                                              Mesh Utils
# ----------------------------------------------------------------------------------------------------------------------#
def calc_vnrmls(v, f):
    # NOTE - Vertices unreferenced by faces will be zero
    a = v[f[:, 0], :]
    b = v[f[:, 1], :]
    c = v[f[:, 2], :]
    fn = np.cross(b - a, c - a)
    matrix = index_sparse(v.shape[0], f)
    vn = matrix.dot(fn)
    return normr(vn)


def padded_part_by_mask(mask_vi, gt_v):
    # Pad the mask to length:
    needed_padding_len = gt_v.shape[0] - len(mask_vi)
    mask_vi_padded = np.append(mask_vi, np.random.choice(mask_vi, needed_padding_len, replace=True))  # Copies
    return gt_v[mask_vi_padded,:]


def calc_moments(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)

# ----------------------------------------------------------------------------------------------------------------------#
#                                              Unchecked
# ----------------------------------------------------------------------------------------------------------------------#

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
    vn = F.normalize(vn, p=2, dim=1)
    # vn = normr(vn)
    # Old Vertex Normals
    # vn = np.zeros_like(v)
    # vn[self.ref_tri[:, 0], :] = vn[self.ref_tri[:, 0], :] + fn
    # vn[self.ref_tri[:, 1], :] = vn[self.ref_tri[:, 1], :] + fn
    # vn[self.ref_tri[:, 2], :] = vn[self.ref_tri[:, 2], :] + fn

    return vn


def calc_vnrmls_batch(v, f_tup):
    # v dimensions: [batch_size x 3 x n_vertices]
    # f dimensions: ( [n_faces x 3] , [n_faces x 3] )
    v = v.transpose(2, 1)
    vn = torch.zeros_like(v)
    for i in range(v.shape[0]):
        vn[i, :, :] = calc_vnrmls_torch(v[i, :, :], f_tup)

    v = v.transpose(2, 1)
    vn = vn.transpose(2, 1)
    return vn

    # XF = V[:, :, triv].transpose(2,
    #                              1)  # first dimension runs on the vertices in the triangle, second on the triangles and third on x,y,z coordinates
    # N = torch.cross(XF[:, :, :, 1] - XF[:, :, :, 0],
    #                 XF[:, :, :, 2] - XF[:, :, :, 0])  # OH: normal field T x 3, directed outwards
    # N = N / torch.sqrt(torch.sum(N ** 2, dim=-1, keepdim=True))


def calc_euclidean_dist_matrix(x):
    # OH: x contains the coordinates of the mesh,
    # x dimensions are [batch_size x num_nodes x 3]

    x = x.transpose(2, 1)
    r = torch.sum(x ** 2, dim=2).unsqueeze(2)  # OH: [batch_size  x num_points x 1]
    r_t = r.transpose(2, 1)  # OH: [batch_size x 1 x num_points]
    inner = torch.bmm(x, x.transpose(2, 1))
    D = F.relu(r - 2 * inner + r_t) ** 0.5  # OH: the residual numerical error can be negative ~1e-16
    return D

def test_normals(v, f, n):
    # test_normals(template, self.ref_tri, template_n)
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=0.2, antialiased=True)
    vnn = v + n
    ax.quiver(v[:, 0], v[:, 1], v[:, 2], vnn[:, 0], vnn[:, 1], vnn[:, 2], length=0.03, normalize=True)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   Graveyard
# ----------------------------------------------------------------------------------------------------------------------#

# def calc_vnrmls(v, f):
#     # NOTE - Vertices unreferenced by faces will be zero
#     # Compute Face Normals
#     a = v[f[:, 0], :]
#     b = v[f[:, 1], :]
#     c = v[f[:, 2], :]
#     fn = np.cross(b - a, c - a)
#
#     # Compute Vertex Normals
#     matrix = index_sparse(v.shape[0], f)
#     vn = matrix.dot(fn)
#     # Normalize them
#     # Note - in some runs I've made, vectors computed are degenrate and cause errors in the computation.
#     # The normr function masks these - I.I.
#     # vn = vn / np.sqrt(np.sum(vn ** 2, -1, keepdims=True)) # Does not handle 0 vectors
#     vn = normr(vn)
#     # Old Vertex Normals
#     # vn = np.zeros_like(v)
#     # vn[self.ref_tri[:, 0], :] = vn[self.ref_tri[:, 0], :] + fn
#     # vn[self.ref_tri[:, 1], :] = vn[self.ref_tri[:, 1], :] + fn
#     # vn[self.ref_tri[:, 2], :] = vn[self.ref_tri[:, 2], :] + fn
