import argparse
import fnmatch
import os
from progress.bar import Bar
from data_prep.external_tools.pyRender.src import gen_projections
import re

read_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST/'
save_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Projections/MPI-FAUST/'
error_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/error_files.txt'

for root, dirnames, filenames in os.walk(read_path):
    for filename in fnmatch.filter(filenames, '*.obj*'):
        output_dirname = re.sub('\.obj$', '', os.path.join(save_path, root.replace(read_path, save_path), filename))
        gen_projections.run(os.path.join(root, filename), output_dirname, error_path, scale=100)

