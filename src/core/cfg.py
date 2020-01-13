import numpy as np
from pathlib import Path
import torch
# from datetime import datetime
# TODO - See if we can get rid of this file
# ----------------------------------------------------------------------------------------------------------------------
#                                              COMPUTATION CONFIG
# ----------------------------------------------------------------------------------------------------------------------

# Use in loss.py & datasets.py
DEF_CPU_PRECISION = np.float32  # Translates to tensor types as well
DEF_GPU_PRECISION = torch.float32
# Used in transforms.py
DANGEROUS_MASK_THRESH = 100
NORMAL_MAGNITUDE_THRESH = 10 ** (-10)


# Used in pytorch_extensions.py
RANDOM_SEED = 2147483647  # datetime.now() For a truly random seed


# ----------------------------------------------------------------------------------------------------------------------
#                                                  DATASET CONFIG
# ----------------------------------------------------------------------------------------------------------------------

# Used in abstract.py & datasets.py
PRIMARY_RESULTS_DIR = Path(__file__).parents[0] / '..' / '..' / 'results'
PRIMARY_DATA_DIR = Path(__file__).parents[0] / '..' / '..' / 'data'
SUPPORTED_IN_CHANNELS = (3, 6, 12)



