#!/bin/bash
#SBATCH -p all # partition (queue).
#SBATCH -N 1 # number of nodes
#SBATCH -n 8
#SBATCH --gres=gpu:01
#SBATCH -w xeon-09
#SBATCH --job-name=train
#SBATCH --output=results_train.txt
#SBATCH --error=errors.txt

conda init bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate continuum-rl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

python DDPG.py