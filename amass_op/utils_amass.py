import glob
import torch
from torch.utils.data import Dataset
import os


def show_image(img_ndarray, id):

    '''
    Visualize images resulted from calling vis_smpl_params in Jupyternotebook
    :param img_ndarray: Nx400x400x3
    '''

    import matplotlib as plt
    import os
    import numpy as np
    import cv2

    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.gca()

    img = img_ndarray.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    plt.axis('off')

    if not os.path.isdir("images"):
        os.makedirs("images")
    fig_name = "images/fig" + str(id) + ".png"
    plt.savefig(fig_name)
    plt.close()
    # fig.canvas.draw()
    # return True


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


def write_off(name, body, faces):

    with open(name, "w") as f:
        f.write("OFF")
        f.write("\n")
        f.write(str(len(body.v[0])))
        f.write(" ")
        f.write(str(len(faces)))
        f.write(" ")
        f.write("0")
        f.write("\n")
        for t in body.v[0]:
            for j in t.data:
                f.write(str(j.item()))
                f.write(" ")
            f.write("\n")

        for g in faces:
            f.write("3")
            f.write(" ")
            for k in g:
                f.write(str(k))
                f.write(" ")
            f.write("\n")


amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
              'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
}

amass_splits_ids = {

    "vald": [3, 7, 14, 33],
    "test": [1, 4],
    "train": [96, 99, 104, 116, 171, 282, 286, 287, 307]
}
