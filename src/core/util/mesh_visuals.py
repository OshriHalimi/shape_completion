import torch
import numpy as np
import pyvista as pv
from util.mesh_compute import face_barycenters
from vtkplotter.actors import Actor
from vtkplotter.utils import buildPolyData


# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Visualization Functions
# ----------------------------------------------------------------------------------------------------------------------#
def show_point_cloud(v, n=None, grid_on=True):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param n: (optional): vertex normals (numpy array or tensors), dim: [n_v x 3]
    :param grid_on: With/without grid
    :return: plot point cloud with vertex normals (if provided)
    """

    vertices = v.numpy() if torch.is_tensor(v) else v
    normals = n.numpy() if torch.is_tensor(n) else n

    p = pv.Plotter()
    point_cloud = pv.PolyData(vertices)
    p.add_mesh(point_cloud, color=[0, 1, 0])
    if n is not None:
        point_cloud['vectors'] = normals
        arrows = point_cloud.glyph(scale=False, factor=0.03, )
        p.add_mesh(arrows, color=[0, 0, 1])
    if grid_on:
        p.show_grid()
    p.show()


def show_vnormals(v, f, n=None):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :param n: vertex normals (numpy array or tensors), dim: [n_v x 3]
    :return: plot mesh with vertex normals
    """
    v = v.numpy() if torch.is_tensor(v) else v
    f = f.numpy() if torch.is_tensor(f) else f
    n = n.numpy() if torch.is_tensor(n) else n

    p = pv.Plotter()
    if n is not None:
        point_cloud = pv.PolyData(v)
        point_cloud['vectors'] = n
        arrows = point_cloud.glyph(scale=False, factor=0.03, )
        p.add_mesh(arrows, color=[0, 0, 1])

    f = np.concatenate((np.full((f.shape[0], 1), 3), f), 1)
    surface = pv.PolyData(v, f)
    p.add_mesh(surface, color="grey", ambient=0.6, opacity=1, show_edges=True)
    p.show()


def show_fnormals(v, f, n=None):
    """
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param f: faces (numpy array or tensors), dim: [n_f x 3]
    :param n: face normals (numpy array or tensors), dim: [n_v x 3]
    :return: plot mesh with face normals
    """
    c = face_barycenters(v, f)
    c = c.numpy() if torch.is_tensor(c) else c
    v = v.numpy() if torch.is_tensor(v) else v
    f = f.numpy() if torch.is_tensor(f) else f
    n = n.numpy() if torch.is_tensor(n) else n

    p = pv.Plotter()
    if n is not None:
        point_cloud = pv.PolyData(c)
        point_cloud['vectors'] = n
        arrows = point_cloud.glyph(scale=False, factor=0.03, )
        p.add_mesh(arrows, color=[0, 0, 1])

    f = np.concatenate((np.full((f.shape[0], 1), 3), f), 1)
    surface = pv.PolyData(v, f)
    p.add_mesh(surface, color="grey", ambient=0.6, opacity=1, show_edges=True)
    p.show()


def show_vnormals_matplot(v, f, n):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=1, antialiased=True)
    ax.quiver(v[:, 0], v[:, 1], v[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.03, normalize=True)
    ax.set_aspect('equal', 'box')
    plt.show()


def show_fnormals_matplot(v, f, n):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    c = face_barycenters(v, f)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, linewidth=1, antialiased=True)
    ax.quiver(c[:, 0], c[:, 1], c[:, 2], n[:, 0], n[:, 1], n[:, 2], length=0.03, normalize=True)
    ax.set_aspect('equal', 'box')
    plt.show()


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
