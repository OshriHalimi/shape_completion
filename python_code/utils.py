import random
import numpy as np
import torch
import torch.nn.functional as F
import sys

try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def calc_euclidean_dist_matrix(x):
    #OH: x contains the coordinates of the mesh,
    #x dimensions are [batch_size x num_nodes x 3]

    x = x.transpose(2,1)
    r = torch.sum(x ** 2, dim=2).unsqueeze(2)  # OH: [batch_size  x num_points x 1]
    r_t = r.transpose(2, 1) # OH: [batch_size x 1 x num_points]
    inner = torch.bmm(x,x.transpose(2, 1))
    D = F.relu(r - 2 * inner + r_t)**0.5  # OH: the residual numerical error can be negative ~1e-16
    return D

#def apply_different_rotation_for_each_point(R,x):
    # OH: R is a torch tensor of dimensions [batch x num_points x 3 x 3]
    #     x i a torch tensor of dimenstions [batch x num_points x 3    ]
    #     the result has the same dimensions as x

#initialize the weighs of the network for Convolutional layers and batchnorm layers
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

