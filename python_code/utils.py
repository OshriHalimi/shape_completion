import random
import numpy as np
import torch
import torch.nn.functional as F
import sys
from plyfile import PlyData, PlyElement
import scipy
from sklearn.preprocessing import normalize
# ----------------------------------------------------------------------------------------------------------------------#
#                                               Neural Network Utis
# ----------------------------------------------------------------------------------------------------------------------#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  # OH: n is the weight corresponding to the current value val
        self.val = val
        self.sum += val * n  # OH: weighted sum
        self.count += n  # OH: sum of weights
        self.avg = self.sum / self.count  # OH: weighted average


def read_lr(path_to_log):
    epoch_list = []
    epoch = 0
    with open(path_to_log, "r") as f:

        for line in f:

            if line.startswith("EPOCH NUMBER"):
                line = line.strip("\n")
                line = line.split(":")
                epoch_list.append(int(line[-1]))
    if len(epoch_list) > 0:
        epoch = max(epoch_list)
    return epoch


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


def index_sparse(columns, indices, data=None):
    """
    Return a sparse matrix for which vertices are contained in which faces.
    A data vector can be passed which is then used instead of booleans
    """
    indices = np.asanyarray(indices)
    columns = int(columns)
    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape((-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    if data is None:
        data = np.ones(len(col), dtype=np.bool)
    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)), shape=shape, dtype=data.dtype)

    return matrix


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
#                                              Mesh File Utils
# ----------------------------------------------------------------------------------------------------------------------#
def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

# def apply_different_rotation_for_each_point(R,x):
# OH: R is a torch tensor of dimensions [batch x num_points x 3 x 3]
#     x i a torch tensor of dimenstions [batch x num_points x 3    ]
#     the result has the same dimensions as x

# initialize the weighs of the network for Convolutional layers and batchnorm layers


def normc(Mat):
    return normalize(Mat, norm='l2', axis=0)

def normr(Mat):
    return normalize(Mat, norm='l2', axis=1)

def normv(Vec):
    return normalize(Vec, norm='l2')
