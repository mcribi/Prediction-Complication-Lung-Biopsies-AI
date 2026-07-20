#!/bin/bash
#SBATCH -J gradcam
#SBATCH -p dgx
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err
#SBATCH -c 8
#SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc


cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta

/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python posthoc_gradcam_multilabel.py