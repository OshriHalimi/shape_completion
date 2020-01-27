import pathlib
# from datetime import datetime
""" SHORTCUTS TABLE 
r = reconstruction
b = batch
v = vertex
s = string or split 
vn = vertex normals
f = face
fn = face normals 
gt = ground truth
tp = template
i = index 
fp = file path 
dp = directory path 
hp = hyper parameters 
ds = dataset 
You can also concatenate - gtrb = Ground Truth Reconstruction Batched  
"""
# ----------------------------------------------------------------------------------------------------------------------
#                                              COMPUTATION CONFIG
# ----------------------------------------------------------------------------------------------------------------------

# TIP: Use CTRL + SHIFT + f in PyCharm to detect where these are used in the system

RANDOM_SEED = 2147483647  # The global random seed. Use datetime.now() For a truly random seed
#TODO: there is bug in DEF_COMPUTE_PRECISION: changing to float64 results in a crash
DEF_COMPUTE_PRECISION = 'float32' # Default computation precision used through out the entire system
NORMAL_MAGNITUDE_THRESH = 10 ** (-6) # The minimal norm allowed for vertex normals to decide that they are too small

# Hyper parameters that were removed from main.py
DEF_LR_SCHED_COOLDOWN = 5 # Number of epoches to wait after reducing the step-size. Works only if LR sched is enabled
DEF_MINIMAL_LR = 1e-6 # The smallest learning step allowed with LR sched. Works only if LR sched is enabled
NON_BLOCKING = True # Transfer to GPU in a non-blocking method
REPORT_LOSS_PER_BATCH = False # If True - will output train loss to logger on every batch. Otherwise - on every epoch
MAX_EPOCHS = 1000
# ----------------------------------------------------------------------------------------------------------------------
#                                               PATH CONFIG
# ----------------------------------------------------------------------------------------------------------------------

PRIMARY_RESULTS_DIR = (pathlib.Path(__file__).parents[0] / '..' / '..' / 'results').resolve()
PRIMARY_DATA_DIR = (pathlib.Path(__file__).parents[0] / '..' / '..' / 'data').resolve()

# ----------------------------------------------------------------------------------------------------------------------
#                                               ERROR CONFIG
# ----------------------------------------------------------------------------------------------------------------------
SUPPORTED_IN_CHANNELS = (3, 6, 12) # The possible supported input channels - either 3, 6 or 12
DANGEROUS_MASK_THRESH = 100 # The minimal length allowed for mask vertex indices.


