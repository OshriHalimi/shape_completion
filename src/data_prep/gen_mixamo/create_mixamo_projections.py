import subprocess
import argparse
import fnmatch
import os
from progress.bar import Bar
from data_prep.external_tools.pyRender import gen_projections
import re

read_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Blender/MPI-FAUST/'
save_path = '/run/user/1000/gvfs/smb-share:server=132.68.36.59,share=data/ShapeCompletion/Mixamo/Projections/MPI-FAUST/'

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', default = read_path, help='absolute path to input folder')
parser.add_argument('--output_folder', default = save_path, help='absolute path to output folder')
parser.add_argument('--max_files', help='maximum number of files to process at a time (if it is too big it is going to run oom)', default=1000, type=int)



args = parser.parse_args()

for root, dirnames, filenames in os.walk(args.input_folder):
    for filename in fnmatch.filter(filenames, '*.obj*'):
        output_dirname = re.sub('\.obj$', '', os.path.join(args.output_folder, root.replace(read_path, save_path), filename))
        gen_projections.run([os.path.join(root, filename)], output_dirname)



