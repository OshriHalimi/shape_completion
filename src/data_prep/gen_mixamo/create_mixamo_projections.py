import subprocess
import argparse
import fnmatch
import os
from progress.bar import Bar
from data_prep.external_tools.pyRender import gen_projections
import re
import pyrender
import trimesh
import pathlib
from core.util.mesh.plot import plot_mesh
import matplotlib.pyplot as plt
import numpy as np

# def gen_projections(input_filename, output_dirname):
#     pc = trimesh.load(open(input_filename,'r'), 'obj')
#     V = np.array(pc.vertices)
#     F = np.array(pc.faces)
#     bbox_x = [np.min(V[:, 0]), np.max(V[:, 0])]
#     bbox_y = [np.min(V[:, 1]), np.max(V[:, 1])]
#     bbox_z = [np.min(V[:, 2]), np.max(V[:, 2])]
#     center = 0.5 * np.array([bbox_x[0] + bbox_x[1], bbox_y[0] + bbox_y[1], bbox_z[0] + bbox_z[1]])
#     V = V - np.expand_dims(center, axis=0)
#     plot_mesh(V, F, strategy='mesh', grid_on=True)
#
#     # vertices: a Nx3 double numpy array
#     # faces: a Nx3 int array (indices of vertices array)
#     # cam_intr: (fx, fy, px, py) double vector
#     # img_size: (height, width) int vector
#     cam_intr = (575, 575, 319.5, 239.5)
#     img_size = (480,640)
#     depth, mask = pyrender.render(V, F, cam_intr, img_size)
#     if len(mask) == 1:
#         print("WARNING: WRONG LENGTH !!!", file)
#
#     print(out_path)
#     out_name = os.path.join(out_path, f"{str(i_ang)}")
#     pathlib.Path(out_name).mkdir(parents=True, exist_ok=True)


read_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST/'
save_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Projections/MPI-FAUST/'

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', default = read_path, help='absolute path to input folder')
parser.add_argument('--output_folder', default = save_path, help='absolute path to output folder')
parser.add_argument('--max_files', help='maximum number of files to process at a time (if it is too big it is going to run oom)', default=1000, type=int)
args = parser.parse_args()

for root, dirnames, filenames in os.walk(args.input_folder):
    for filename in fnmatch.filter(filenames, '*.obj*'):
        output_dirname = re.sub('\.obj$', '', os.path.join(args.output_folder, root.replace(read_path, save_path), filename))
        gen_projections.run([os.path.join(root, filename)], output_dirname)

