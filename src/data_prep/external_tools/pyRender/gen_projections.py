def run(in_path, out_path):
	import numpy as np
	import sys
	import os
	import time
	import trimesh
	import pathlib
	from core.util.mesh.plot import plot_mesh
	import matplotlib.pyplot as plt

	libpath = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(libpath + '/lib')
	import render

	# these need to be full_paths
	for file in in_path:

		# t = time.time()

		pc = trimesh.load(file,'obj')
		V = np.array(pc.vertices)*100
		F = np.array(pc.faces)

		# set up camera information,
		info = {'Height':480, 'Width':640, 'fx':575, 'fy':575, 'cx':319.5, 'cy':239.5}
		render.setup(info)

		# set up mesh buffers in cuda
		context = render.SetMesh(V, F)
		cam2world = np.array([[ 0.85408425,  0.31617427, -0.375678  ,  0.56351697 * 2],
			   [ 0.        , -0.72227067, -0.60786998,  0.91180497 * 2],
			   [-0.52013469,  0.51917219, -0.61688   ,  0.92532003 * 2],
			   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)


		# TODO: add options for more elevations
		# rotate the mesh elevation by 30 degrees
		Rx = np.array([[ 1, 0, 0, 0],
				   [ 0.        , np.cos(np.pi/6), -np.sin(np.pi/6), 0],
				   [0, np.sin(np.pi/6),  np.cos(np.pi/6), 0],
				   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)
		cam2world = np.matmul(Rx,cam2world)

		# TODO: add option for different angles
		# rotate along y axis and render
		for i_ang, ang in enumerate(np.linspace(0,2*np.pi, 10)):

			Ry = np.array([[ np.cos(ang),  0, -np.sin(ang)  ,  0],
				   [ 0.        , 1, 0,  0],
				   [np.sin(ang),  0, np.cos(ang)   ,  0],
				   [ 0.        ,  0.        ,  0.        ,  1.        ]], dtype=np.float32)
			world2cam = np.linalg.inv(np.matmul(Ry,cam2world)).astype('float32')


			# the actual rendering process
			render.render(context, world2cam)

			# get depth information
			depth = render.getDepth(info)
			#plt.imshow(depth)

			# get information of mesh rendering
			# vindices represents 3 vertices related to pixels
			# vweights represents barycentric weights of the 3 vertices
			# findices represents the triangle index related to pixels
			vindices, vweights, findices = render.getVMap(context, info)
			mask = np.unique(vindices)
			if len(mask) == 1:
				print("WARNING: WRONG LENGTH !!!", file)

			output_path = os.path.join(os.getcwd(), "data_output_train")
			if not os.path.exists(output_path):
				os.makedirs(output_path)

			print(f"{os.path.split(file)[1][:-4]}_{str(i_ang)}")
			print(out_path)
			out_name = os.path.join(out_path, f"{str(i_ang)}")
			pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

			#plot_mesh(V[mask,:], strategy='spheres', grid_on=True)

			np.savez(out_name, mask=mask)
			render.Clear()
			# print(f"time: {time.time() - t}")

if __name__ == "__main__":
	main()





