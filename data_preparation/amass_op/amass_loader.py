import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import trimesh
from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.visualization_tools import imagearray2file
from human_body_prior.body_model.body_model import BodyModel
from utils_amass import show_image
from bisect import bisect_left
from utils_amass import AMASS_DS
from utils_amass import write_off
from utils_amass import amass_splits
from utils_amass import amass_splits_ids

if not os.path.isdir("amass_dump"):
    os.mkdir("amass_dump")

expr_code = 'V1_S1_T1'  # VERSION_SUBVERSION_TRY
work_dir = os.path.join(os.getcwd(), expr_code)
num_betas = 16  # number of body parameters
num_dmpls = 8   # numner of dmpls paramters
batch_size = 1
# path to the body models
# can be downloaded at http://mano.is.tue.mpg.de/
bm_path = os.path.join(os.getcwd(), expr_code, "body_models", "smplh", "male", "model.npz")
# can be downloaded at http://smpl.is.tue.mpg.de/downloads
dmpl_path = os.path.join(os.getcwd(), expr_code, "body_models", "dmpls", "male", "model.npz")

bm = BodyModel(bm_path=bm_path, num_betas=num_betas,
               path_dmpl=dmpl_path, num_dmpls=num_dmpls, batch_size=batch_size)  # .to(comp_device)
faces = c2c(bm.f)

# Choose the device to run the body model on.
#comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for fold in ["train", "test", "vald"]:

    path_split_dump = os.path.join("amass_dump", fold)
    if not os.path.isdir(path_split_dump):
        os.mkdir(path_split_dump)

    path_split_dump_images = os.path.join("amass_dump", fold + "_images")
    if not os.path.isdir(path_split_dump_images):
        os.mkdir(path_split_dump_images)

    split_dir = os.path.join(work_dir, 'stage_III', fold)
    print(split_dir)

    ds = AMASS_DS(dataset_dir=split_dir, num_betas=num_betas)
    print(" {} split has {} datapoints.".format(fold, len(ds)))

    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)

    old_betas = torch.ones([1, 16])
    sub_id = 0
    frame_id = 0
    for i, bdata in enumerate(dataloader):

        root_orient = bdata["root_orient"]#.to(comp_device)
        pose_body = bdata["pose_body"]#.to(comp_device)
        pose_hand = bdata["pose_hand"]#.to(comp_device)
        betas = bdata["betas"]#.to(comp_device)
        dmpls = bdata["dmpl"]#.to(comp_device)

        body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, dmpls=dmpls)

        new_betas = betas
        if not torch.equal(new_betas, old_betas):
            sub_id += 1
            frame_id = 0

        dataset_name_id = bisect_left(amass_splits_ids[fold], sub_id)
        dataset_name = amass_splits[fold][dataset_name_id]

        dump_name = "Dataset_" + str(dataset_name) + "_" + "subjectID_" + str(sub_id) + "_" + "Frame_" + str(frame_id)
        dump_name_off = dump_name + ".OFF"
        dump_path_off = os.path.join(path_split_dump, dump_name_off)
        write_off(dump_path_off, body, faces)

        # plot
        dump_name_png = dump_name + ".png"
        dump_path_png = os.path.join(path_split_dump_images, dump_name_png)
        b = body.v[0].detach().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(b[:, 2], b[:, 0], b[:, 1])
        plt.savefig(dump_path_png)
        plt.close()

        old_betas = new_betas
        frame_id += 1

        print("FINISHED SAMPLE: ", i)





    # body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, dmpls=dmpls)

    # break
    #
    # imw, imh = 1600, 1600
    # mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    #
    # body_mesh_wdmpls = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces,
    #                                    vertex_colors=np.tile(colors['grey'], (6890, 1)))


    # f = open("prova.txt", "w")
    # f.write("OFF")
    # f.write("\n")
    # f.write(str(len(body.v[0])))
    # f.write(" ")
    # f.write(str(len(faces)))
    # f.write(" ")
    # f.write("0")
    # f.write("\n")
    # for t in body.v[0]:
    #     for j in t.data:
    #         f.write(str(j.item()))
    #         f.write(" ")
    #     f.write("\n")
    #
    # for g in faces:
    #     f.write("3")
    #     f.write(" ")
    #     for k in g:
    #         f.write(str(k))
    #         f.write(" ")
    #     f.write("\n")
    # f.close()

    # p = trimesh.points.PointCloud(body.v[0].detach().numpy())
    # p.save("prova")





    # mv.set_static_meshes([body_mesh_wdmpls])
    # body_image_wdmpls = mv.render(render_wireframe=False)
    # show_image(body_image_wdmpls, i)
    # print("DOING SAMPLE #: ", i)
