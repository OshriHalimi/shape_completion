import numpy as np
import sys
import skimage.io as sio
import os
import sys
import shutil
from objloader import LoadTextureOBJ
from core.util.mesh.plot import plot_mesh
from core.util.mesh.io import read_off
libpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(libpath + '/../lib')
import render
import objloader
import matplotlib.pyplot as plt
import trimesh

#input_obj = './../resources/occlude.obj'
input_obj = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST/000/Au/010.obj'
#input_obj = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/subjects/tr_reg_000.obj'
mesh = trimesh.load(input_obj)
V = mesh.vertices * 100
F = mesh.faces

#input_off = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/data/synthetic/FaustPyProj/full/tr_reg_000.off'
#input_off = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/data/synthetic/AmassTestPyProj/full/subjectID_1_poseID_0.OFF'
#input_off = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/data/synthetic/DFaustPyProj/full/50002/chicken_wings/00001.OFF'
#V, F = read_off(input_off)

V = V.astype(np.float32)
F = F.astype(np.int32)
#V, F = objloader.LoadOff(input_off)

bbox_x = [np.min(V[:,0]), np.max(V[:,0])]
bbox_y = [np.min(V[:,1]), np.max(V[:,1])]
bbox_z = [np.min(V[:,2]), np.max(V[:,2])]
center = 0.5 * np.array([bbox_x[0] + bbox_x[1], bbox_y[0] + bbox_y[1], bbox_z[0] + bbox_z[1]])
V = V - np.expand_dims(center, axis = 0)



#N = mesh.vertex_normals
plot_mesh(V, F, strategy='mesh', grid_on=True)


# if input_obj[-3:] == 'obj':
# 	V, F, VT, FT, VN, FN, face_mat, kdmap = objloader.LoadTextureOBJ(input_obj)
# else:
# 	V, F = objloader.LoadOff(input_obj)

print(np.min(V), np.max(V))

# set up camera information',
info = {'Height': 480, 'Width': 640, 'fx': 575, 'fy': 575, 'cx': 319.5, 'cy': 239.5}
render.setup(info)

# set up mesh buffers in cuda
context = render.SetMesh(V, F)

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
	depth = render.getDepth(info)
	plt.imshow(depth)


	# get information of mesh rendering
	# vindices represents 3 vertices related to pixels
	# vweights represents barycentric weights of the 3 vertices
	# findices represents the triangle index related to pixels
	vindices, vweights, findices = render.getVMap(context, info)
	mask = np.unique(vindices)
	plot_mesh(V[mask,:], strategy='spheres', grid_on=True)


	print(findices.shape)
	vis_face = findices.astype('float32') / np.max(findices)
	sio.imsave('face.png',vis_face)
	sio.imsave('vertex.png',vweights)
