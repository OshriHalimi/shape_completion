import numpy as np
import open3d as o3d
from util.mesh_io import read_off,write_off,write_obj
from dataset.transforms import flip_mask

def tester():
    pass

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


if __name__ == "__main__":
    tester()


# import torch
# import torchvision
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
# from architecture.PytorchNet import PytorchNet
#
# # Writer will output to ./runs/ directory by default
# # writer = SummaryWriter()
#
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# model = torchvision.models.resnet50(False)
# # Have ResNet architecture take in grayscale rather than RGB
# for (x,y) in trainloader:
#     break
# model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# pymodel = PytorchNet.monkeypatch(model)
# pymodel.print_weights()
# print(pymodel.family_name())
# # pymodel.visualize()
#
# print(pymodel.output_size())
# pymodel.summary()
