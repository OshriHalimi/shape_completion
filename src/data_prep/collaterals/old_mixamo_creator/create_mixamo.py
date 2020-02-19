import numpy as np
import trimesh
import os
import pathlib
import fnmatch
import re
from external.PyRender.lib import render


def run(filename, out_dir, error_path, scale, cam2world, info):
    pc = trimesh.load(filename, 'obj')
    V = np.array(pc.vertices) * scale
    F = np.array(pc.faces)
    bbox_x = [np.min(V[:, 0]), np.max(V[:, 0])]
    bbox_y = [np.min(V[:, 1]), np.max(V[:, 1])]
    bbox_z = [np.min(V[:, 2]), np.max(V[:, 2])]
    center = 0.5 * np.array([bbox_x[0] + bbox_x[1], bbox_y[0] + bbox_y[1], bbox_z[0] + bbox_z[1]])
    V = V - np.expand_dims(center, axis=0)
    V = V.astype(np.float32)
    F = F.astype(np.int32)

    render.setup(info)
    # set up mesh buffers in cuda
    context = render.set_mesh(V, F)
    # TODO: add option for different angles
    # rotate along y axis and render
    for i_ang, ang in enumerate(np.linspace(0, 2 * np.pi, 10)):

        Ry = np.array([[np.cos(ang), 0, -np.sin(ang), 0],
                       [0., 1, 0, 0],
                       [np.sin(ang), 0, np.cos(ang), 0],
                       [0., 0., 0., 1.]], dtype=np.float32)
        world2cam = np.linalg.inv(np.matmul(Ry, cam2world)).astype('float32')

        # the actual rendering process
        render.render(context, world2cam)

        # get depth information
        # depth = render.getDepth(info)
        # plt.imshow(depth)

        # get information of mesh rendering
        # vindices represents 3 vertices related to pixels
        # vweights represents barycentric weights of the 3 vertices
        # findices represents the triangle index related to pixels
        vindices, vweights, findices = render.get_vmap(context, info)
        mask = np.unique(vindices)
        if len(mask) == 1:
            print("ERROR: WRONG MASK !!!", filename)
            if not os.path.isfile(error_path):
                open(error_path, "x")
            f = open(error_path, "a")
            f.write(filename + "\n")
            f.close()
            break

        out_name = os.path.join(out_dir, f"{str(i_ang)}")
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        # plot_mesh(V[mask,:], strategy='spheres', grid_on=True)

        np.savez(out_name, mask=mask)
    # print(out_name)

    render.clear()


def mixamo_main():
    # from data_prep.external_tools.pyRender.src import gen_projections

    read_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST/000/'
    save_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Projections/MPI-FAUST/000/'
    error_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/error_files_000.txt'

    # set up camera information,
    info = {'Height': 480, 'Width': 640, 'fx': 575, 'fy': 575, 'cx': 319.5, 'cy': 239.5}
    cam2world = np.array([[0.85408425, 0.31617427, -0.375678, 0.56351697 * 2],
                          [0., -0.72227067, -0.60786998, 0.91180497 * 2],
                          [-0.52013469, 0.51917219, -0.61688, 0.92532003 * 2],
                          [0., 0., 0., 1.]], dtype=np.float32)

    # rotate the mesh elevation by 30 degrees
    Rx = np.array([[1, 0, 0, 0],
                   [0., np.cos(np.pi / 6), -np.sin(np.pi / 6), 0],
                   [0, np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                   [0., 0., 0., 1.]], dtype=np.float32)
    cam2world = np.matmul(Rx, cam2world)

    count = 0
    flag = True
    for root, dirnames, filenames in os.walk(read_path):
        for filename in fnmatch.filter(filenames, '*.obj*'):
            if not flag:
                if os.path.join(root,
                                filename) == '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST/050/Catwalk Walk/041.obj':
                    flag = True
                    print("Reached the first crashed file!")
            if flag:
                output_dirname = re.sub('\.obj$', '',
                                        os.path.join(save_path, root.replace(read_path, save_path), filename))
                run(os.path.join(root, filename), output_dirname, error_path, scale=100, cam2world=cam2world, info=info)
            count += 1
            print(count)
