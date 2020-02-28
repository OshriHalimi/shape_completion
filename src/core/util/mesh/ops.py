import torch
import numpy as np
import torch.nn.functional as F
from util.matrix import normr, index_sparse
import cfg


# ----------------------------------------------------------------------------------------------------------------------#
#                                    Singleton Computes - Hybrid Numpy & PyTorch
# ----------------------------------------------------------------------------------------------------------------------#

def face_barycenters(v, f):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :return: face_centers (numpy array or tensors), dim: [n_f x 3]
    """
    return (1 / 3) * (v[f[:, 0], :] + v[f[:, 1], :] + v[f[:, 2], :])


# ----------------------------------------------------------------------------------------------------------------------#
#                                       Singleton Computes for Numpy Only
# ----------------------------------------------------------------------------------------------------------------------#
def fnrmls(v, f, normalized=True):
    # TODO - Deprecated if proven slow vs Oshri's version
    a = v[f[:, 0], :]
    b = v[f[:, 1], :]
    c = v[f[:, 2], :]
    fn = np.cross(b - a, c - a)
    if normalized:
        fn = normr(fn)
    return fn


def vnrmls(v, f, normalized=True):
    # TODO - Deprecated if proven slow vs Oshri's version
    # NOTE - Vertex normals unreferenced by faces will be zero
    if f is None:
        raise NotImplementedError  # TODO - Add in computation for scans, without faces - either with pcnormals/
    else:
        fn = fnrmls(v, f, normalized=False)
        matrix = index_sparse(v.shape[0], f)
        vn = matrix.dot(fn)
        if normalized:
            vn = normr(vn)
        return vn


def moments(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)


def padded_part_by_mask(vi, v):
    # Pad the mask to length:
    needed_padding_len = v.shape[0] - len(vi)  # Truncates ALL input channels
    mask_vi_padded = np.append(vi, np.random.choice(vi, needed_padding_len, replace=True))  # Copies
    return v[mask_vi_padded, :]


def flip_vertex_mask(nv, vi):
    indicator = vertex_mask_indicator(nv, vi)
    return np.where(indicator == 0)[0]


def vertex_mask_indicator(nv, vi):
    indicator = np.zeros((nv,), dtype=bool)
    indicator[vi] = 1
    return indicator


def trunc_to_vertex_mask(v, f, vi):
    # TODO: Warning supports only watertight meshes (not scans) - Need to remove vertices unref by f2
    if f is None:
        return v[vi, :], None
    else:
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

        # TODO - Check this
        # faces = faces.reshape(-1)
        # unique_points_index = np.unique(faces)
        # unique_points = pts[unique_points_index]

    return v2, f2


def box_center(v):
    bbox_x = [np.min(v[:, 0]), np.max(v[:, 0])]
    bbox_y = [np.min(v[:, 1]), np.max(v[:, 1])]
    bbox_z = [np.min(v[:, 2]), np.max(v[:, 2])]
    center = 0.5 * np.array([bbox_x[0] + bbox_x[1], bbox_y[0] + bbox_y[1], bbox_z[0] + bbox_z[1]])
    return v - np.expand_dims(center, axis=0)

# TODO - Check this
def normalize_unitL2ball_pointcloud(points):
    #normalize  to unit ball pointcloud
    #points N_points, 3
    points[:,0:3] = points[:,0:3] / np.sqrt(np.max(np.sum(points[:,0:3]**2, 1)))
    return points

# TODO - Check this
def normalize_by_channel(points):
    #normalize  to unit ball pointcloud
    #points N_points, 3
    points[:,0] = points[:,0] / np.max(points[:,0])
    points[:,1] = points[:,1] / np.max(points[:,1])
    points[:,2] = points[:,2] / np.max(points[:,2])
    return points


# ----------------------------------------------------------------------------------------------------------------------#
#                                       Singleton Computes for Torch Only
# ----------------------------------------------------------------------------------------------------------------------#
# TODO: validate me
def calc_volume(v, f):
    v1 = v[:, f[:, 0], :]
    v2 = v[:, f[:, 1], :]
    v3 = v[:, f[:, 2], :]
    a_vec = torch.cross(v2 - v1, v3 - v1, -1)
    center = (v1 + v2 + v3) / 3
    volume = torch.sum(a_vec * center / 6, dim=(1, 2))
    return volume


def vf_adjacency(faces, n_faces, n_verts, device):
    """
    :param faces: dim: [N_faces x 3]
    :param n_faces: number of faces
    :param n_verts: number of vertices
    :param device: device to place tensors
    :return: adjacency_vf: sparse integer adjacency matrix between vertices and faces, dim: [N_vertices x N_faces]
    """
    fvec = torch.arange(n_faces, device=device)
    i0 = torch.stack((faces[:, 0], fvec), dim=1)
    i1 = torch.stack((faces[:, 1], fvec), dim=1)
    i2 = torch.stack((faces[:, 2], fvec), dim=1)
    ind = torch.cat((i0, i1, i2), dim=0)
    ones_vec = torch.ones([3 * n_faces], dtype=torch.int8, device=device)
    adjacency_vf = torch.sparse.IntTensor(ind.t(), ones_vec, torch.Size([n_verts, n_faces]))
    return adjacency_vf


# ----------------------------------------------------------------------------------------------------------------------#
#                                       PyTorch Batch Computations
# ----------------------------------------------------------------------------------------------------------------------#
def batch_euclid_dist_mat(vb):
    # vb of dim: [batch_size x nv x 3]
    r = torch.sum(vb ** 2, dim=2, keepdim=True)  # [batch_size  x num_points x 1]
    inner = torch.bmm(vb, vb.transpose(2, 1))
    return F.relu(r - 2 * inner + r.transpose(2, 1)) ** 0.5  # the residual numerical error can be negative ~1e-16


def batch_moments(vb):
    # TODO - Implement
    # x, y, z = v[:, 0], v[:, 1], v[:, 2]
    # return np.stack((x ** 2, y ** 2, z ** 2, x * y, x * z, y * z), axis=1)
    raise NotImplementedError


def batch_fnrmls_fareas(vb, f, return_normals=True):
    """ # TODO - Allow also [n_verts x 3]. Write another function called batch_fnrmls if we only need those
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param f: faces matrix,we assume all the shapes have the same connectivity, dim: [n_faces x 3], dtype = torch.long
    :param return_normals : Whether to return the normals or not
    :return face_normals_b: batch of face normals, dim: [batch_size x n_faces x 3]
            face_areas_b: batch of face areas, dim: [batch_size x n_faces x 1]
            is_valid_fnb: boolean matrix indicating if the normal is valid,
            magnitude greater than zero [batch_size x n_faces].
            If the normal is not valid we return [0,0,0].
    """

    # calculate xyz coordinates for 1-3 vertices in each triangle
    v1 = vb[:, f[:, 0], :]  # dim: [batch_size x n_faces x 3]
    v2 = vb[:, f[:, 1], :]  # dim: [batch_size x n_faces x 3]
    v3 = vb[:, f[:, 2], :]  # dim: [batch_size x n_faces x 3]

    face_normals_b = torch.cross(v2 - v1, v3 - v2)
    face_areas_b = torch.norm(face_normals_b, dim=2, keepdim=True) / 2
    if not return_normals:
        return face_areas_b

    is_valid_fnb = (face_areas_b.squeeze(2) > (cfg.NORMAL_MAGNITUDE_THRESH / 2))
    fnb_out = torch.zeros_like(face_normals_b)
    fnb_out[is_valid_fnb, :] = face_normals_b[is_valid_fnb, :] / (2 * face_areas_b[is_valid_fnb, :])
    return fnb_out, face_areas_b, is_valid_fnb


def batch_vnrmls(vb, f, return_f_areas=False):
    """
    :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
    :param f: faces matrix, here we assume all the shapes have the same sonnectivity, dim: [n_faces x 3]
    :param return_f_areas: Whether to return the face areas or not
    :return: vnb:  batch of shape normals, per vertex, dim: [batch_size x n_vertices x 3]
    :return: is_valid_vnb: boolean matrix indicating if the normal is valid, magnitude greater than zero
    [batch_size x n_vertices].
    If the normal is not valid we return [0,0,0].
    :return face_areas_b (optional): a batch of face areas, dim: [batch_size x n_faces x 1]
    """

    n_faces = f.shape[0]
    n_batch = vb.shape[0]

    face_normals_b, face_areas_b, is_valid_fnb = batch_fnrmls_fareas(vb, f)
    # non valid face normals are: [0, 0, 0], due to batch_fnrmls_fareas
    face_normals_b *= face_areas_b  # weight each normal with the corresponding face area

    face_normals_b = face_normals_b.repeat(1, 3, 1)  # repeat face normals 3 times along the face dimension
    f = f.t().contiguous().view(3 * n_faces)  # dim: [n_faces x 3] --> [(3*n_faces)]
    f = f.expand(n_batch, -1)  # dim: [B x (3*n_faces)]
    f = f.unsqueeze(2).expand(n_batch, 3 * n_faces, 3)
    # dim: [B x (3*n_faces) x 3], last dimension (xyz dimension) is repeated

    # For each vertex, sum all the normals of the adjacent faces (weighted by their areas)
    vnb = torch.zeros_like(vb)  # dim: [batch_size x n_vertices x 3]
    vnb = vnb.scatter_add(1, f, face_normals_b)  # vb[b][f[b,f,xyz][xyz] = face_normals_b[b][f][xyz]

    magnitude = torch.norm(vnb, dim=2, keepdim=True)
    is_valid_vnb = (magnitude > cfg.NORMAL_MAGNITUDE_THRESH).squeeze(2)
    vnb_out = torch.zeros_like(vb)
    vnb_out[is_valid_vnb, :] = vnb[is_valid_vnb, :] / magnitude[is_valid_vnb, :]
    # check the sum of face normals is greater than zero

    return (vnb_out, is_valid_vnb, face_areas_b) if return_f_areas else (vnb_out, is_valid_vnb)


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Test Suite
# ----------------------------------------------------------------------------------------------------------------------

def bring_in_test_data():
    from dataset.datasets import FullPartDatasetMenu
    from dataset.transforms import Center
    ds = FullPartDatasetMenu.get('FaustPyProj')
    samp = ds.sample(num_samples=5, transforms=[Center()], method='f2p')  # dim:
    vb = samp['gt'][:, :, :3]
    f = torch.from_numpy(ds.faces()).long()
    return vb, f


def test_vnrmls_grad():
    vb, f = bring_in_test_data()
    # N_faces = faces.shape[0]
    # N_vertices = batch_v.shape[1]
    # adjacency_vf = vf_adjacency(faces, N_faces, N_vertices)
    # This operation can be calculated once for the whole training

    from torch.autograd import gradcheck
    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    x = (vb.requires_grad_(True).double(), f)
    test = gradcheck(batch_vnrmls, x, eps=1e-6, atol=1e-4, check_sparse_nnz=True)
    print(test)


def test_vnrmls_visually():
    from util.mesh.plots import plot_mesh
    vb, f = bring_in_test_data()
    # adjacency_VF = vf_adjacency(faces, n_faces, n_verts)
    # This operation can be calculated once for the whole training
    vertex_normals, is_valid_vnb = batch_vnrmls(vb, f)
    # There exist 2 implementations for batch_vnrmls, batch_vnrmls_ uses adjacency_VF while batch_vnrmls doesn'r
    # magnitude = torch.norm(vertex_normals, dim=2)  # Debug: assert the values are equal to 1.000

    v = vb[4, :, :]
    n = vertex_normals[4, :, :]
    plot_mesh(v, f, n)


def test_fnrmls_visually():
    from util.mesh.plots import plot_mesh
    vb, f = bring_in_test_data()
    fn, is_valid_fnb, face_areas_b = batch_fnrmls_fareas(vb, f)
    # magnitude = torch.norm(face_normals, dim=2)  # Debug: assert the values are equal to 1.000
    v = vb[4, :, :]
    n = fn[4, :, :]
    plot_mesh(v, f, n)


if __name__ == '__main__':
    test_vnrmls_visually()

# ----------------------------------------------------------------------------------------------------------------------
#                                        Graveyard
# ----------------------------------------------------------------------------------------------------------------------

# This implementation for batch vertex normals is here since currently it consumes too much memory
# and pytorch doesn't support sparse matrix bmm
# def batch_vnrmls_(vb, f, adj_vf):
#     """.
#     :param vb: batch of shape vertices, dim: [batch_size x n_vertices x 3]
#     :param f: faces matrix, here we assume all the shapes have the same sonnectivity, dim: [n_faces x 3]
#     :param adj_vf: sparse adjacency matrix beween vertices and faces, dim: [n_vertices x n_faces]
#     :return: vnb:  batch of shape normals, per vertex, dim: [batch_size x n_vertices x 3]
#     :return: is_valid_vnb: boolean matrix indicating if the normal is valid,
#     magnitude greater than zero [batch_size x n_vertices].
#     """
#     PytorchNet.print_memory_usage()
#     face_normals_b, face_areas_b, is_valid_fnb = batch_fnrmls_fareas(vb, f)
#     is_valid_fnb = is_valid_fnb.unsqueeze(1)
#     face_areas_b = face_areas_b.unsqueeze(1)
#     adj_vf = adj_vf.unsqueeze(0)
#
#     weights_vf = is_valid_fnb * face_areas_b * adj_vf.to_dense()  # dim: [batch_size x n_vertices x n_faces]
#     total_weight_v = torch.norm(weights_vf, dim=2, keepdim=True, p=1)
#     weights_vf = weights_vf / total_weight_v
#     total_weight_v = total_weight_v.squeeze(2)
#     is_valid_vnb = total_weight_v > 0  # A check that at least one valid face contributes to the average
#     vnb = weights_vf.bmm(face_normals_b)  # face_normals_b dim: [batch_size x n_faces x 3]
#
#     magnitude = torch.norm(vnb, dim=2, keepdim=True)
#     is_valid_vnb = is_valid_vnb * (magnitude.squeeze(2) > cfg.NORMAL_MAGNITUDE_THRESH)
#     vnb_out = torch.zeros_like(vb)
#     vnb_out[is_valid_vnb, :] = vnb[is_valid_vnb, :] / magnitude[is_valid_vnb, :]
#     # A check that the average normal is greater than zero
#
#     return vnb, is_valid_vnb
