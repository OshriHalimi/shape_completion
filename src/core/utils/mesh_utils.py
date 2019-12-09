import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from data_utils import normr, index_sparse

# ----------------------------------------------------------------------------------------------------------------------#
#                                              Mesh Utils
# ----------------------------------------------------------------------------------------------------------------------#
def calc_vnrmls(v, f):
    # NOTE - Vertices unreferenced by faces will be zero
    # Compute Face Normals
    a = v[f[:, 0], :]
    b = v[f[:, 1], :]
    c = v[f[:, 2], :]
    fn = np.cross(b - a, c - a)

    # Compute Vertex Normals
    matrix = index_sparse(v.shape[0], f)
    vn = matrix.dot(fn)
    # Normalize them
    # Note - in some runs I've made, vectors computed are degenrate and cause errors in the computation.
    # The normr function masks these - I.I.
    # vn = vn / np.sqrt(np.sum(vn ** 2, -1, keepdims=True)) # Does not handle 0 vectors
    vn = normr(vn)
    # Old Vertex Normals
    # vn = np.zeros_like(v)
    # vn[self.ref_tri[:, 0], :] = vn[self.ref_tri[:, 0], :] + fn
    # vn[self.ref_tri[:, 1], :] = vn[self.ref_tri[:, 1], :] + fn
    # vn[self.ref_tri[:, 2], :] = vn[self.ref_tri[:, 2], :] + fn

    return vn

def calc_vnrmls_torch(v, f_tup):
    f,f_torch = f_tup

    a = v[f_torch[:, 0], :]
    b = v[f_torch[:, 1], :]
    c = v[f_torch[:, 2], :]
    fn = torch.cross(b - a, c - a)

    matrix = index_sparse(v.shape[0], f)
    matrix  = torch.from_numpy(matrix.todense()).float().cuda()
    vn = torch.mm(matrix,fn)
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
    v = v.transpose(2,1)
    vn = torch.zeros_like(v)
    for i in range(v.shape[0]):
        vn[i, :, :] = calc_vnrmls_torch(v[i, :, :], f_tup)

    v = v.transpose(2, 1)
    vn = vn.transpose(2,1)
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

# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Unit Tests
# ----------------------------------------------------------------------------------------------------------------------#

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