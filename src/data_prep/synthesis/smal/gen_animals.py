import os
from smpl.smpl_webuser.serialization import load_model
from smal.my_mesh.mesh import myMesh as mesh_loader
import pickle as pkl
import numpy as np
from tqdm import tqdm
import sys

"""
# Load the family clusters data (see paper for details)
# and save the mean per-family shape
# 0-felidae(cats); 1-canidae(dogs); 2-equidae(horses);
# 3-bovidae(cows); 4-hippopotamidae(hippos);
# The clusters are over the shape coefficients (betas);
# setting different betas changes the shape of the model
"""

SMAL_MODEL = load_model(os.path.join('smal', 'smal_CVPR2017.pkl'))
SMAL_DATA = pkl.load(open(os.path.join('smal', 'smal_CVPR2017_data.pkl'), "rb"))
OUTPUT_DIR = os.path.join('.', 'outputs')
NULL_POSE_DIR = os.path.join(OUTPUT_DIR, 'null_pose')
os.makedirs(NULL_POSE_DIR)

NUM_TRAIN_PER_SUBJECT = 5000
NUM_VALD_PER_SUBJECT = 1000
NUM_TEST_PER_SUBJECT = 1000

for i, (betas, sub_name) in enumerate(zip(SMAL_DATA['cluster_means'], ['cats', 'dogs', 'horses', 'cows', 'hippos'])):

    SMAL_MODEL.betas[:] = betas
    SMAL_MODEL.pose[:] = 0.
    SMAL_MODEL.trans[:] = 0.

    m = mesh_loader(v=SMAL_MODEL.r, f=SMAL_MODEL.f)
    m.save_ply(os.path.join(NULL_POSE_DIR, sub_name + '_null.ply'))

    train_dir = os.path.join(OUTPUT_DIR, 'train', sub_name)
    os.makedirs(train_dir)
    vald_dir = os.path.join(OUTPUT_DIR, 'vald', sub_name)
    os.makedirs(vald_dir)
    test_dir = os.path.join(OUTPUT_DIR, 'test', sub_name)
    os.makedirs(test_dir)

    # TEST DATA
    for i in tqdm(range(NUM_TEST_PER_SUBJECT), file=sys.stdout, dynamic_ncols=True):
        SMAL_MODEL.pose[0:3] = 0
        SMAL_MODEL.pose[3:] = 0.2 * np.random.randn(96)
        m = mesh_loader(v=SMAL_MODEL.r, f=SMAL_MODEL.f)
        m.save_ply(os.path.join(test_dir, str(i) + '.ply'))

    # TEST DATA
    for i in tqdm(range(NUM_VALD_PER_SUBJECT), file=sys.stdout, dynamic_ncols=True):
        SMAL_MODEL.pose[0:3] = 0
        SMAL_MODEL.pose[3:] = 0.2 * np.random.randn(96)
        m = mesh_loader(v=SMAL_MODEL.r, f=SMAL_MODEL.f)
        m.save_ply(os.path.join(vald_dir, str(i) + '.ply'))

    # TRAINING DATA
    for i in tqdm(range(NUM_TRAIN_PER_SUBJECT), file=sys.stdout, dynamic_ncols=True):
        SMAL_MODEL.pose[0:3] = 0
        SMAL_MODEL.pose[3:] = 0.2 * np.random.randn(96)
        m = mesh_loader(v=SMAL_MODEL.r, f=SMAL_MODEL.f)
        m.save_ply(os.path.join(train_dir, str(i) + '.ply'))
