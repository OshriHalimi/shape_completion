import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from data_utils import normr, index_sparse
import os

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

def read_npz_mask(fp):
    return np.load(fp)["mask"]

def read_off_verts(fp):
    vbuf = []
    with open(fp, "r") as f:
        first = f.readline().strip()
        if first != "OFF":
            raise (Exception(f"Could not find OFF header for file: {first}"))

        parameters = f.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception(f"Wrong number of parameters fount at OFF file: {first}"))

        for i in range(int(parameters[0])):
            xyz = f.readline().split()
            vbuf.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    return np.array(vbuf)


def read_off_full(fp):
    vbuf = []
    fbuf = []
    with open(fp, "r") as f:
        first = f.readline().strip()
        if first != "OFF":
            raise (Exception(f"Could not find OFF header for file: {first}"))

        parameters = f.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception(f"Wrong number of parameters fount at OFF file: {first}"))

        for i in range(int(parameters[0])):
            xyz = f.readline().split()
            vbuf.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

        for i in range(int(parameters[1])):
            inds = f.readline().split()
            fbuf.append([int(inds[1]), int(inds[2]), int(inds[3])])

    return np.array(vbuf), np.array(fbuf)
