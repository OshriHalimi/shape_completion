import torch
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------#
#                                                    Visualization Functions
# ----------------------------------------------------------------------------------------------------------------------#
def show_point_cloud(v, n=None, grid_on=True):
    '''
    :param v: vertices (numpy array or tensors), dim: [n_v x 3]
    :param n (optional): vertex normals (numpy array or tensors), dim: [n_v x 3]
    :return: plot point cloud with vertex normals (if provided)
    '''
    vertices = v.numpy() if torch.is_tensor(v) else v
    normals = n.numpy() if torch.is_tensor(n) else n

    import pyvista as pv
    p = pv.Plotter()
    point_cloud = pv.PolyData(vertices)
    p.add_mesh(point_cloud, color=[0, 1, 0])
    if n is not None:
            point_cloud['vectors'] = normals
            arrows = point_cloud.glyph(orient='vectors', scale=False, factor=0.03, )
            p.add_mesh(arrows, color=[0, 0, 1])
    if grid_on:
        p.show_grid()
    p.show()


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
