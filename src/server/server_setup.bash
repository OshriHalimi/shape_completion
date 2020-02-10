#!/bin/bash 
#-------------------------------------------- On your first time: Manually --------------------------------------------#
# git clone https://github.com/OshriHalimi/shape_completion.git shape_completion 
# chmod 777 ~/shape_completion/src/server/*
# Add source ~/shape_completion/src/server/server_setup.bash to ~/.bashrc
# pip3 install --user --upgrade pip
# pip3 install --user torch torchvision
# pip3 install --user -r ~/shape_completion/requirements.txt
#----------------------------------------------------------------------------------------------------------------------#
# Basic Tutorial: 
# 8 Gaon Servers, ~4 GPUs per Server
# Best Server are: 8->2,3->4,5,6,7. Gaon1 is CPU only
# Max Jobs - Between 3 & 5 
# Server IP:
SERVER_IP=`hostname -I | sed 's/ *$//g'`
echo "Use $SERVER_IP:6006 to log-on tensorboard"

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


