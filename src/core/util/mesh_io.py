import numpy as np
from vtkplotter.actors import Actor
from vtkplotter.utils import buildPolyData


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


def read_off(fp):
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

def write_off(fp, v, f=None):
    if f is None:
        f = []
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    str_f = [f"3 {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    with open(fp, 'w') as meshfile:
        meshfile.write(f'OFF\n{len(str_v)} {len(str_f)} 0\n{"".join(str_v)}{"".join(str_f)}')


def write_obj(fp, v, f=None):
    if f is None:
        f = []
    else:
        f += 1 # Faces are 1-based, not 0-based in obj files
    str_v = [f"v {vv[0]} {vv[1]} {vv[2]}\n" for vv in v]
    str_f = [f"f {ff[0]} {ff[1]} {ff[2]}\n" for ff in f]
    with open(fp, 'w') as meshfile:
        meshfile.write(f'{"".join(str_v)}{"".join(str_f)}')


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   VTK Platform
# ----------------------------------------------------------------------------------------------------------------------#

def numpy2vtkactor(v, f, clr='gold'):
    return Actor(buildPolyData(v, f, ), computeNormals=False, c=clr)  # Normals are in C++ - Can't extract them


def print_vtkplotter_help():
    print("""
==========================================================
| Press: i     print info about selected object            |
|        m     minimise opacity of selected mesh           |
|        .,    reduce/increase opacity                     |
|        /     maximize opacity                            |
|        w/s   toggle wireframe/solid style                |
|        p/P   change point size of vertices               |
|        l     toggle edges line visibility                |
|        x     toggle mesh visibility                      |
|        X     invoke a cutter widget tool                 |
|        1-3   change mesh color                           |
|        4     use scalars as colors, if present           |
|        5     change background color                     |
|        0-9   (on keypad) change axes style               |
|        k     cycle available lighting styles             |
|        K     cycle available shading styles              |
|        o/O   add/remove light to scene and rotate it     |
|        n     show surface mesh normals                   |
|        a     toggle interaction to Actor Mode            |
|        j     toggle interaction to Joystick Mode         |
|        r     reset camera position                       |
|        C     print current camera info                   |
|        S     save a screenshot                           |
|        E     export rendering window to numpy file       |
|        q     return control to python script             |
|        Esc   close the rendering window and continue     |
|        F1    abort execution and exit python kernel      |
| Mouse: Left-click    rotate scene / pick actors          |
|        Middle-click  pan scene                           |
|        Right-click   zoom scene in or out                |
|        Cntrl-click   rotate scene perpendicularly        |
|----------------------------------------------------------|
| Check out documentation at:  https://vtkplotter.embl.es  |
 ==========================================================""")


# ----------------------------------------------------------------------------------------------------------------------#
#                                                   TODO - Integrate
# ----------------------------------------------------------------------------------------------------------------------#

def write_ply(fp, v, f, n, clrs):
    str_vertices = ["{} {} {}".format(v[0], v[1], v[2]) for v in v]
    str_indices = ["3 {} {} {}\n".format(i[0], i[1], i[2]) for i in f]
    str_normals = ["{} {} {}".format(n[0], n[1], n[2]) for n in n]
    # no transparency, alpha = 255
    str_colors = ["{} {} {}".format(c[0], c[1], c[2]) for c in clrs]

    str_vertices = ["{} {} {}\n".format(str_vertices[i], str_normals[i], str_colors[i]) for i in range(len(v))]

    with open(fp, "w") as meshfile:
        meshfile.write('''ply
format ascii 1.0
comment VCGLIB generated
element vertex {0}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face {1}
property list uchar int vertex_indices
end_header
{2}
{3}
'''.format(len(str_vertices), len(str_indices), ''.join(str_vertices), ''.join(str_indices)))

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   GRAVEYARD
# ----------------------------------------------------------------------------------------------------------------------#
# def mesh_montage():
#     fig, axs = plt.subplots(montage_shape[0], montage_shape[1], subplot_kw={'projection': '3d'}, sharex='all',
#                             sharey='all')
#     # plt.show()
#     # figManager = plt.get_current_fig_manager()
#     # figManager.full_screen_toggle()
#     for i, ax in enumerate(axs.flat):
#         ax.view_init(elev=90, azim=270)
#         v = samp[i][ind]
#         ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=self._f, linewidth=0.2, antialiased=True)
#         ax.axis('image')
#         ax.axis('off')
#         # vnn = v + n
#         # ax.quiver(v[:, 0], v[:, 1], v[:, 2], vnn[:, 0], vnn[:, 1], vnn[:, 2], length=0.03, normalize=True)
#     # fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
#     # fig.tight_layout(pad=0)
#     # fig.subplots_adjust(wspace = 0,hspace=0)
#     plt.show()
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
# def test_normals(v, f, n):
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=0.2, antialiased=True)
#     vnn = v + n
#     ax.quiver(v[:, 0], v[:, 1], v[:, 2], vnn[:, 0], vnn[:, 1], vnn[:, 2], length=0.03, normalize=True)
#     plt.show()

# ----------------------------------------------------------------------------------------------------------------------#
#                                                   GRAVEYARD - VISDOM
# ----------------------------------------------------------------------------------------------------------------------#
#     if opt.use_visdom:
#         vis = visdom.Visdom(port=8888, env=opt.save_path)
#             # VIZUALIZE
#             if opt.use_visdom and i % 100 == 0:
#                 vis.scatter(X=part[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Part',
#                             opts=dict(title="Train_Part", markersize=2, ), )
#                 vis.scatter(X=template[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Template',
#                             opts=dict(title="Train_Template", markersize=2, ), )
#                 vis.scatter(X=gt_rec[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_output',
#                             opts=dict(title="Train_output", markersize=2, ), )
#                 vis.scatter(X=gt[0, :3, :].transpose(1, 0).contiguous().data.cpu(), win='Train_Ground_Truth',
#                             opts=dict(title="Train_Ground_Truth", markersize=2, ), )
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
#                      Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val))),
#                      win='Faust loss',
#                      opts=dict(title="Faust loss", legend=["Train loss", "Faust Validation loss", ]))
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val)))),
#                      Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val)))),
#                      win='"Faust log loss',
#                      opts=dict(title="Faust log loss", legend=["Train loss", "Faust Validation loss", ]))
#
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
#                      Y=np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass))),
#                      win='AMASS loss',
#                      opts=dict(title="AMASS loss", legend=["Train loss", "Validation loss", "Validation loss amass"]))
#             vis.line(X=np.column_stack((np.arange(len(Loss_curve_train)), np.arange(len(Loss_curve_val_amass)))),
#                      Y=np.log(np.column_stack((np.array(Loss_curve_train), np.array(Loss_curve_val_amass)))),
#                      win='AMASS log loss',
#                      opts=dict(title="AMASS log loss", legend=["Train loss", "Faust Validation loss", ]))
#
