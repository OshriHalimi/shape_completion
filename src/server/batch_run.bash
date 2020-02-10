#!/bin/bash 

#SBATCH -N 1 # number of minimum nodes 
#SBATCH -c 2 # number of cores 
#SBATCH –gres=gpu:1   # Request 1 gpu
#SBATCH -p gip
#SBATCH -w gaon2
#SBATCH –mail-user=ido.imanuel@gmail.com # oshri.halimi@gmail.com 
#SBATCH –mail-type=ALL  # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="ShapeCompletionBatch" 
#SBATCH -o slurm.%N.%j.out # stdout goes here 
#SBATCH -e slurm.%N.%j.out # stderr goes here
python3 ~/shape_completion/src/core/main.py

# Command: sbatch batch_run.bash
# tail -f ~/slurm.NodeName.JobID.out (fill in your NodeName,JobID)