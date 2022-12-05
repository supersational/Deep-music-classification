#!/usr/bin/env bash

#SBATCH --job-name=lab4
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./slogs/log_bc4_%j.out # STDOUT out
#SBATCH -e ./slogs/log_bc4_%j.err # STDERR out
#SBATCH --gres=gpu:1
#SBATCH --time=0:01:00
#SBATCH --mem=16GB

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"
# conda activate sven

python -u main.py >> "./slogs/log_bc4_$SLURM_JOB_ID.txt"

# find /user/home/sh14603/Deep-music-classification/src/ -type f -mmin -10 -exec tail -f {} +
