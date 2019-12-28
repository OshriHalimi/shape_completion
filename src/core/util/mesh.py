import numpy as np
from plyfile import PlyData, PlyElement


# TODO - Add visualizations here
# ----------------------------------------------------------------------------------------------------------------------#
#                                                   READ
# ----------------------------------------------------------------------------------------------------------------------#
def read_npz_mask(fp):
    return np.load(fp)["mask"]


def read_off_verts(fp):
    vbuf = []
    with open(fp, "r") as f:
        first = f.readline().strip()
        if first != "OFF" and first != "COFF":
            raise (Exception(f"Could not find OFF header for file: {fp}"))

        parameters = f.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

        for i in range(int(parameters[0])):
            xyz = f.readline().split()
            vbuf.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

    return np.array(vbuf)


def read_off_full(fp):
    vbuf = []
    fbuf = []
    with open(fp, "r") as f:
        first = f.readline().strip()
        if first != "OFF" and first != "COFF":
            raise (Exception(f"Could not find OFF header for file: {fp}"))

        parameters = f.readline().strip().split()

        if len(parameters) < 2:
            raise (Exception(f"Wrong number of parameters fount at OFF file: {fp}"))

        for i in range(int(parameters[0])):
            xyz = f.readline().split()
            vbuf.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

        for i in range(int(parameters[1])):
            inds = f.readline().split()
            fbuf.append([int(inds[1]), int(inds[2]), int(inds[3])])

    return np.array(vbuf), np.array(fbuf)


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   WRITE
# ----------------------------------------------------------------------------------------------------------------------#

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   GRAVEYARD
# ----------------------------------------------------------------------------------------------------------------------#
# def mesh_montage(images, cls_true, label_names, cls_pred=None, siz=3):
#     # Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
#     fig, axes = plt.subplots(siz, siz)
#
#     for i, ax in enumerate(axes.flat):
#         # plot img
#         ax.imshow(images[i, :, :, :].squeeze(), interpolation='spline16', cmap='gray')
#
#         # show true & predicted classes
#         cls_true_name = label_names[cls_true[i]]
#         if cls_pred is None:
#             xlabel = f"{cls_true_name} ({cls_true[i]})"
#         else:
#             cls_pred_name = label_names[cls_pred[i]]
#             xlabel = f"True: {cls_true_name}\nPred: {cls_pred_name}"
#         ax.set_xlabel(xlabel)
#         ax.set_xticks([])
#         ax.set_yticks([])
#
#     plt.show()
