import numpy as np
import open3d as o3d
from util.mesh_io import read_off, write_off, write_obj
from util.gen import list_class_declared_methods, list_narrow_class_methods, list_dynasty_class_methods, \
    list_parent_class_methods
from dataset.transforms import flip_mask
import torchvision
from architecture.PytorchNet import PytorchNet
from types import FunctionType
import inspect


class Parent:
    PARENT_STATIC = 1

    def __init__(self):
        self.father_inside = 5

    def papa(self):
        pass

    def mama(self):
        pass

    @classmethod
    def parent_class(cls):
        pass

    @staticmethod
    def parent_static():
        pass


class Son(Parent):
    SON_VAR = 1

    def __init__(self):
        super().__init__()
        self.son_inside = 1

    def papa(self):
        pass

    def child(self):
        pass

    @classmethod
    def son_class(cls):
        pass

    @staticmethod
    def son_static():
        pass


# -----------------------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------------------
from architecture.PytorchNet import PytorchNet
from timeit import timeit
from pprint import pprint


def tester():
    n = 11
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i + j >= n - 1 and j % 2 == 0:
                mat[i][j] = 1

    pprint(mat)

    # print(sorted(list(list_dynasty_class_methods(Son))))
    # print(sorted(list(list_parent_class_methods(Son))))
    # print(sorted(list(list_class_declared_methods(Son))))
    # print(sorted(list(list_narrow_class_methods(Son))))


# -----------------------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------------------
def voxels():
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("frag.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    # o3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    # o3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10, :])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
    chair = vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair])
    print("")

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([chair])
    print("")


# -----------------------------------------------------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    tester()
