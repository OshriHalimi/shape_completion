import torch
import numpy as np
import torch.nn.functional as F
from util.datascience import normr, index_sparse
from torch_scatter import scatter_add
from util.gen import warn
import random
import cfg


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
        # del x['f'] # Remove this as an optimization
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
#                                               Test Functions
# ----------------------------------------------------------------------------------------------------------------------

def test_vnrmls_grad():
    from dataset.datasets import PointDatasetMenu,InCfg
    ds = PointDatasetMenu.get('FaustPyProj',in_channels=12,in_cfg=InCfg.FULL2PART)
    samp = ds.sample(num_samples=2, transforms=[Center()]) #dim:
    batch_v = samp['gt'][:,:,:3]
    batch_f = samp['f']
    batch_f = batch_f.long()
    faces = batch_f[0,:,:]
    N_faces = faces.shape[0]
    N_vertices = batch_v.shape[1]

    adjacency_VF = calc_adjacency_VF(faces, N_faces, N_vertices)  # This operation can be calculated once for the whole training

    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (batch_v.requires_grad_(True).double(), faces)
    test = gradcheck(batch_vnrmls_2, input, eps=1e-6, atol=1e-4, check_sparse_nnz=True)
    print(test)


def test_vnrmls_visually():
    from dataset.datasets import PointDatasetMenu,InCfg
    ds = PointDatasetMenu.get('FaustPyProj',in_channels=12,in_cfg=InCfg.FULL2PART)
    samp = ds.sample(num_samples=10, transforms=[Center()]) #dim:
    batch_v = samp['gt'][:,:,:3]
    batch_f = samp['f']
    batch_f = batch_f.long()
    faces = batch_f[0,:,:]
    N_faces = faces.shape[0]
    N_vertices = batch_v.shape[1]

    adjacency_VF = calc_adjacency_VF(faces, N_faces, N_vertices)  # This operation can be calculated once for the whole training
    vertex_normals, is_valid_vnb = batch_vnrmls(batch_v, faces)  # There exist 2 implementations for batch_vnrmls, batch_vnrmls_ uses adjacency_VF while batch_vnrmls doesn'r
    magnitude = torch.norm(vertex_normals, dim = 2)  # Debug: assert the values are equal to 1.000

    v = batch_v[4,:,:]
    f = faces
    n = vertex_normals[4,:,:]
    show_vnormals(v, f, n)
    print(samp)

def test_fnrmls_visually():
    from dataset.datasets import PointDatasetMenu,InCfg
    ds = PointDatasetMenu.get('FaustPyProj',in_channels=12,in_cfg=InCfg.FULL2PART)
    samp = ds.sample(num_samples=10, transforms=[Center()]) #dim:
    batch_v = samp['gt'][:,:,:3]
    batch_f = samp['f']
    batch_f = batch_f.long()
    faces = batch_f[0,:,:]

    face_normals, is_valid_fnb, face_areas_b = batch_fnrmls_fareas(batch_v, faces)
    magnitude = torch.norm(face_normals, dim=2)  # Debug: assert the values are equal to 1.000

    v = batch_v[4,:,:]
    f = faces
    n = face_normals[4,:,:]
    show_fnormals(v, f, n)
    print(samp)

# ----------------------------------------------------------------------------------------------------------------------
#                                               Data Augmentation Transforms
# ----------------------------------------------------------------------------------------------------------------------

class RandomMaskFlip(Transform):
    def __init__(self, prob):  # Probability of mask flip
        self._prob = prob

    def __call__(self, x):
        if random.random() < self._prob:
            nv = x['gt'].shape[0]
            x['gt_mask_vi'] = flip_mask(nv, x['gt_mask_vi'])
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


# ----------------------------------------------------------------------------------------------------------------------#
#                                           Transform Mesh Functions
# ----------------------------------------------------------------------------------------------------------------------#
def calc_adjacency_VF(faces, N_faces, N_vertices):
    '''
    :param faces: dim: [N_faces x 3]
    :param N_faces: number of faces
    :param N_vertices: number of vertices
    :return: adjacency_VF: sparse integer adjacency matrix between vertices and faces, dim: [N_vertices x N_faces]
    '''
    i0 = torch.stack((faces[:,0], torch.arange(N_faces)), dim=1)
    i1 = torch.stack((faces[:,1], torch.arange(N_faces)), dim=1)
    i2 = torch.stack((faces[:,2], torch.arange(N_faces)), dim=1)
    ind = torch.cat((i0, i1, i2), dim = 0)
    ones_vec = torch.ones([3 * N_faces], dtype=torch.int8)
    adjacency_VF = torch.sparse.IntTensor(ind.t(), ones_vec, torch.Size([N_vertices, N_faces]))
    return adjacency_VF

def calc_face_centers(v, f):
    '''
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: face_centers (numpy array or tensors), dim: [n_f x 3]
    '''
    v1 = v[f[:, 0], :]  # dim: [n_faces x 3]
    v2 = v[f[:, 1], :]  # dim: [n_faces x 3]
    v3 = v[f[:, 2], :]  # dim: [n_faces x 3]

    center_x = (1 / 3) * (v1[:, 0] + v2[:, 0] + v3[:, 0])
    center_y = (1 / 3) * (v1[:, 1] + v2[:, 1] + v3[:, 1])
    center_z = (1 / 3) * (v1[:, 2] + v2[:, 2] + v3[:, 2])

    face_centers = torch.stack((center_x, center_y, center_z), dim = 1) if torch.is_tensor(center_x) else np.stack((center_x, center_y, center_z), axis = 1)
    return face_centers


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
            combined.append(vnrmls(v, f))
        if req_in_channels >= 12 > available_in_channels:
            combined.append(moments(v))

        return np.concatenate(combined, axis=1)


def vnrmls(v, f):
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


def moments(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)


def flip_mask(nv, vi):
    indicator = np.ones((nv,))
    indicator[vi] = 0
    return np.where(indicator == 1)[0]


def trunc_to_vertex_subset(v, f, vi):
    # TODO: Warning supports only watertight meshes (not scans)
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
#                                       PyTorch Batch Computations - TODO - Migrate this
# ----------------------------------------------------------------------------------------------------------------------#
def batch_euclid_dist_mat(vb):
    # vb of dim: [batch_size x nv x 3]
    r = torch.sum(vb ** 2, dim=2, keepdim=True)  # [batch_size  x num_points x 1]
    inner = torch.bmm(vb, vb.transpose(2, 1))  # TODO - Implement correctly
    return F.relu(r - 2 * inner + r.transpose(2, 1)) ** 0.5  # the residual numerical error can be negative ~1e-16


def batch_moments(vb):
    # TODO - Implement
    # x, y, z = v[:, 0], v[:, 1], v[:, 2]
    # return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
    raise NotImplementedError

def batch_fnrmls_fareas(vb, f):
    '''
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param fb: faces matrix, here we assume all the shapes have the same connectivity, dim: [n_faces x 3], dtype = torch.long
    :return face_normals_b: batch of face normals, dim: [batch_size x n_faces x 3]
            face_areas_b: batch of face areas, dim: [batch_size x n_faces]
            is_valid_fnb: boolean matrix indicating if the normal is valid, magnitude greater than zero [batch_size x n_faces]

    Warning: In case the normal magnitude is smaller than a threshold, the normal vector is returned without normalization
    '''

    #calculate xyz coordinates for 1-3 vertices in each triangle
    v1 = vb[:, f[:, 0], :]  # dim: [batch_size x n_faces x 3]
    v2 = vb[:, f[:, 1], :]  # dim: [batch_size x n_faces x 3]
    v3 = vb[:, f[:, 2], :]  # dim: [batch_size x n_faces x 3]

    edge_12 = v2 - v1  # dim: [batch_size x n_faces x 3]
    edge_23 = v3 - v2  # dim: [batch_size x n_faces x 3]

    face_normals_b = torch.cross(edge_12, edge_23)
    face_areas_b = torch.norm(face_normals_b, dim = 2, keepdim=True) / 2

    face_normals_b = face_normals_b / (2 * face_areas_b)
    face_areas_b = face_areas_b.squeeze(2)
    is_valid_fnb = face_areas_b > (cfg.NORMAL_MAGNITUDE_THRESH / 2)

    return face_normals_b, face_areas_b, is_valid_fnb



def batch_vnrmls_(vb, f, adjVF):
    '''
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param f: faces matrix, here we assume all the shapes have the same sonnectivity, dim: [n_faces x 3]
    :param adjVF: sparse adjacency matrix beween vertices and faces, dim: [n_vertices x n_faces]
    :return: vnb:  batch of shape normals, per vertex, dim: [batch_size x n_vertices x 3]
    :return: is_valid_vnb: boolean matrix indicating if the normal is valid, magnitude greater than zero [batch_size x n_vertices]
    '''

    face_normals_b, face_areas_b, is_valid_fnb = batch_fnrmls_fareas(vb, f)
    is_valid_fnb = is_valid_fnb.unsqueeze(1)
    face_areas_b = face_areas_b.unsqueeze(1)
    adjVF = adjVF.unsqueeze(0)

    weights_VF = is_valid_fnb * face_areas_b * adjVF.to_dense() # dim: [batch_size x n_vertices x n_faces]
    total_weight_v = torch.norm(weights_VF, dim=2, keepdim=True, p=1)
    weights_VF = weights_VF / total_weight_v
    total_weight_v = total_weight_v.squeeze(2)
    is_valid_vnb = total_weight_v > 0  # check that at least one valid face contributes to the average
    vnb = weights_VF.bmm(face_normals_b)  # face_normals_b dim: [batch_size x n_faces x 3]

    magnitude = torch.norm(vnb, dim = 2, keepdim=True)
    vnb = vnb / magnitude
    is_valid_vnb = is_valid_vnb * (magnitude.squeeze(2) > cfg.NORMAL_MAGNITUDE_THRESH) # check that the average normal is greater than zero

    return vnb, is_valid_vnb

def batch_vnrmls(vb, f):
    '''
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param f: faces matrix, here we assume all the shapes have the same sonnectivity, dim: [n_faces x 3]
    :return: vnb:  batch of shape normals, per vertex, dim: [batch_size x n_vertices x 3]
    :return: is_valid_vnb: boolean matrix indicating if the normal is valid, magnitude greater than zero [batch_size x n_vertices]
    '''

    N_faces = f.shape[0]
    N_batch = vb.shape[0]

    face_normals_b, face_areas_b, is_valid_fnb = batch_fnrmls_fareas(vb, f)
    face_normals_b[~is_valid_fnb,:] = 0  # non valid face normals --> [0, 0, 0]
    face_normals_b *= face_areas_b.unsqueeze(2)  # weight each normal with the corresponding face area

    face_normals_b = face_normals_b.repeat(1, 3, 1)  # repeat face normals 3 times along the face dimension
    f = f.t().contiguous().view(3 * N_faces)  # dim: [n_faces x 3] --> [(3*n_faces)]
    f = f.expand(N_batch, -1)  # dim: [B x (3*n_faces)]
    f = f.unsqueeze(2).expand(N_batch, 3 * N_faces, 3)  # dim: [B x (3*n_faces) x 3], last dimension (xyz dimension) is repeated

    # For each vertex, sum all the normals of the adjacent faces (weighted by their areas)
    vnb = torch.zeros_like(vb)  # dim: [batch_size x n_vertices x 3]
    vnb = vnb.scatter_add_(1, f, face_normals_b)  # vb[b][f[b,f,xyz][xyz] = face_normals_b[b][f][xyz]

    magnitude = torch.norm(vnb, dim = 2, keepdim=True)
    vnb = vnb / magnitude
    is_valid_vnb = magnitude.squeeze(2) > cfg.NORMAL_MAGNITUDE_THRESH  # check the sum of face normals is greater than zero

    return vnb, is_valid_vnb


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Visualization Functions
# ----------------------------------------------------------------------------------------------------------------------#
def show_vnormals(v, f, n=None):
    '''
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :param n: vertex normals (numpy array or tensors), dim: [n_v x 3]
    :return: plot mesh with vertex normals
    '''
    vertices = v.numpy() if torch.is_tensor(v) else v
    faces = f.numpy() if torch.is_tensor(f) else f
    normals = n.numpy() if torch.is_tensor(n) else n

    import pyvista as pv
    p = pv.Plotter()

    if n is not None:
        point_cloud = pv.PolyData(vertices)
        point_cloud['vectors'] = normals
        arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.03, )
        p.add_mesh(arrows, color=[0, 0, 1])

    faces = np.concatenate((3 * np.ones((f.shape[0], 1)), f), 1)
    surface = pv.PolyData(vertices, faces)
    p.add_mesh(surface, color="grey", ambient=0.6, opacity=1, show_edges=True)
    p.show()

def show_fnormals(v, f, n=None):
    '''
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :param n: face normals (numpy array or tensors), dim: [n_v x 3]
    :return: plot mesh with face normals
    '''
    c = calc_face_centers(v, f)

    centers = c.numpy() if torch.is_tensor(c) else c
    vertices = v.numpy() if torch.is_tensor(v) else v
    faces = f.numpy() if torch.is_tensor(f) else f
    normals = n.numpy() if torch.is_tensor(n) else n

    import pyvista as pv
    p = pv.Plotter()

    if n is not None:
        point_cloud = pv.PolyData(centers)
        point_cloud['vectors'] = normals
        arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.03, )
        p.add_mesh(arrows, color=[0, 0, 1])

    faces = np.concatenate((3 * np.ones((f.shape[0], 1)), f), 1)
    surface = pv.PolyData(vertices, faces)
    p.add_mesh(surface, color="grey", ambient=0.6, opacity=1, show_edges=True)
    p.show()

# matplotlib implementations
def show_vnormals_(v, f, n):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=1, antialiased=True)
    ax.quiver(v[:, 0], v[:, 1], v[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.03, normalize=True)
    ax.set_aspect('equal', 'box')
    plt.show()

def show_fnormals_(v, f, n):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    c = calc_face_centers(v, f)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=1, antialiased=True)
    ax.quiver(c[:, 0], c[:, 1], c[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.03, normalize=True)
    ax.set_aspect('equal', 'box')
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Graveyard
# ----------------------------------------------------------------------------------------------------------------------#
# def calc_vnrmls_torch(v, f_tup):
#     f, f_torch = f_tup
#
#     a = v[f_torch[:, 0], :]
#     b = v[f_torch[:, 1], :]
#     c = v[f_torch[:, 2], :]
#     fn = torch.cross(b - a, c - a)
#
#     matrix = index_sparse(v.shape[0], f)
#     matrix = torch.from_numpy(matrix.todense()).float().cuda()
#     vn = torch.mm(matrix, fn)
#     # Normalize them
#     # Note - in some runs I've made, vectors computed are degenrate and cause errors in the computation.
#     # The normr function masks these - I.I.
#     # vn = vn / np.sqrt(np.sum(vn ** 2, -1, keepdims=True)) # Does not handle 0 vectors
#     vn = tf.normalize(vn, p=2, dim=1)
#     # vn = normr(vn)
#     # Old Vertex Normals
#     # vn = np.zeros_like(v)
#     # vn[self.ref_tri[:, 0], :] = vn[self.ref_tri[:, 0], :] + fn
#     # vn[self.ref_tri[:, 1], :] = vn[self.ref_tri[:, 1], :] + fn
#     # vn[self.ref_tri[:, 2], :] = vn[self.ref_tri[:, 2], :] + fn
#
#     return vn
#
#
# def vnrmls_torch(v, f): # [N x 3]
#     a = v[f[:, 0], :]
#     b = v[f[:, 1], :]
#     c = v[f[:, 2], :]
#     fn = F.normalize(torch.cross(b - a, c - a), p=2, dim=1)
#     vn = scatter_add(fn.repeat(3, 1), f, dim=0, dim_size=v.size(0))
#     return F.normalize(vn, p=2, dim=1)  # [nv, 3]


if __name__ == '__main__': test_fnrmls_visually()
