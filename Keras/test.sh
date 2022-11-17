#!/bin/bash
#SBATCH -p all # partition (queue).
#SBATCH -N 1 # number of nodes
#SBATCH -n 8
#SBATCH --gres=gpu:01
#SBATCH -w xeon-09
#SBATCH --job-name=train_network
#SBATCH --output=results_train.txt
#SBATCH --error=errors.txt

eval "$(conda shell.bash hook)"
conda activate continuum-rl

python DDPG.py