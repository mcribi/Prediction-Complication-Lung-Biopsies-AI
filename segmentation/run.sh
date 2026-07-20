#!/bin/bash
#SBATCH -J pre_mask
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
python ./segmentation_lung_and_nodules.py

