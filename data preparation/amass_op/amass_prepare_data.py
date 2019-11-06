# REQUIRES the amass package, i.e. pip install git+https://github.com/nghorbani/amass
import os
from human_body_prior.tools.omni_tools import makepath, log2file
from amass.prepare_data import prepare_amass

expr_code = 'V1_S1_T1'  # VERSION_SUBVERSION_TRY
msg = ''' Initial use of standard AMASS dataset preparation pipeline '''

# path to the original mocap sequences data
# can be downloaded at https://amass.is.tue.mpg.de/dataset (body data)
amass_dir = os.path.join(os.getcwd(), "data_amass")
# path to the directory where to store the prepared data
work_dir = os.path.join(os.getcwd(), expr_code)

logger = log2file(os.path.join(work_dir, '%s.log' % expr_code))
logger('[%s] AMASS Data Preparation Began.' % expr_code)
logger(msg)

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT',
              'BML', 'EKUT', 'TCD_handMocap', 'ACCAD']
}

# NOTE: BML has to be renamed, CMU's folder has to be rearrenged

prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)
