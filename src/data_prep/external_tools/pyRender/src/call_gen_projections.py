import subprocess
import argparse
import fnmatch
import os
from progress.bar import Bar



parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', help='absolute path to input folder')
parser.add_argument('--output_folder', help='absolute path to output folder')
parser.add_argument('--max_files', help='maximum number of files to process at a time (if it is too big it is going to run oom)', default=1000, type=int)



args = parser.parse_args()

files = []
for root, dirnames, filenames in os.walk(args.input_folder):
    for filename in fnmatch.filter(filenames, '*.*'):
        files.append(os.path.join(root, filename))
files = sorted(files)

if len(files) % args.max_files != 0:
	n_itr = (len(files) // args.max_files) + 1
else:
	n_itr = (len(files) // args.max_files)

bar = Bar('Processing', max=n_itr)
for i in range(n_itr):
	subprocess.call(["python", "gen_projections.py", args.output_folder] + files[(i*args.max_files):((i+1)*args.max_files)])
	bar.next()
bar.finish()



