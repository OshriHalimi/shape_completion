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

expr_code = 'V1_S1_T1'  # VERSION_SUBVERSION_TRY
work_dir = os.path.join(os.getcwd(), expr_code)


class AMASS_DS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir, num_betas = 16):

        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).replace('.pt', '')
            self.ds[k] = torch.load(data_fname)
        self.num_betas = num_betas

    def __len__(self):
        return len(self.ds['trans'])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data['root_orient'] = data['pose'][:3]
        data['pose_body'] = data['pose'][3:66]
        data['pose_hand'] = data['pose'][66:]
        data['betas'] = data['betas'][:self.num_betas]

        return data


# Choose the device to run the body model on.
#comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_betas = 16  # number of body parameters
num_dmpls = 8   # numner of dmpls paramters
testsplit_dir = os.path.join(work_dir, 'stage_III', 'test')
print(testsplit_dir)

ds = AMASS_DS(dataset_dir=testsplit_dir, num_betas=num_betas)
print('Test split has %d datapoints.' % len(ds))

batch_size = 1
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)

# path to the body models
# can be downloaded at http://mano.is.tue.mpg.de/
bm_path = os.path.join(os.getcwd(), expr_code, "body_models", "smplh", "male", "model.npz")
# can be downloaded at http://smpl.is.tue.mpg.de/downloads
dmpl_path = os.path.join(os.getcwd(), expr_code, "body_models", "dmpls", "male", "model.npz")


bm = BodyModel(bm_path=bm_path, num_betas=num_betas,
               path_dmpl=dmpl_path, num_dmpls=num_dmpls, batch_size=batch_size)#.to(comp_device)
faces = c2c(bm.f)

for i, bdata in enumerate(dataloader):

    root_orient = bdata["root_orient"]#.to(comp_device)
    pose_body = bdata["pose_body"]#.to(comp_device)
    pose_hand = bdata["pose_hand"]#.to(comp_device)
    betas = bdata["betas"]#.to(comp_device)
    dmpls = bdata["dmpl"]#.to(comp_device)

    print(betas)


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

    #p = trimesh.points.PointCloud(body.v[0].detach().numpy())
    #p.show()

    # b = body.v[0].detach().numpy()
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(b[:, 2], b[:, 0], b[:, 1])
    # plt.show()
    #plt.savefig("prova.png")

    # mv.set_static_meshes([body_mesh_wdmpls])
    # body_image_wdmpls = mv.render(render_wireframe=False)
    # show_image(body_image_wdmpls, i)
    # print("DOING SAMPLE #: ", i)
