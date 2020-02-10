#!/bin/bash 
#-------------------------------------------- On your first time: Manually --------------------------------------------#
# Add "source ~/shape_completion/src/server/server_setup.bash" to the end of ~/.bashrc
# git clone https://github.com/OshriHalimi/shape_completion.git shape_completion 
# chmod 777 ~/shape_completion/src/server/*
# Install conda via: ~/shape_completion/src/server/Miniconda3-latest-Linux-x86_64.sh 
# Make sure to type 'YES' to all 
# Create a new conda env with: conda create -n shape_completion python==3.7.6
# Activate the environment: conda activate shape_completion
# conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# pip install -r ~/shape_completion/requirements.txt
#----------------------------------------------------------------------------------------------------------------------#
# Basic Tutorial: 
# 8 Gaon Servers, ~4 GPUs per Server
# Best Server are: 8->2,3->4,5,6,7. Gaon1 is CPU only
# Max Jobs - Between 3 & 5 
# Server IP:
SERVER_IP=`hostname -I | sed 's/ *$//g'`
echo "======================================= Server Instructions ========================================"
echo "Surf to $SERVER_IP:6006 using any browser to see tensorboard output"
echo "Use the squeue command to see queue status"
echo "Use the sinfo command to see the server status. down+drain = bad, mix + idle = good"
# echo "===================================================================================================="


# Aliasing: 
alias squeue='squeue -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %Q"'
source /etc/profile.d/bash_completion.sh

# Environment: 
unset XDG_RUNTIME_DIR
export LC_ALL="en_US.UTF-8"
# source /etc/profile.d/modules.sh
# module load cuda
# module load matlab/r2017b
# ../.local/bin/jupyter-lab --no-browser --ip=$hn --port=5698


