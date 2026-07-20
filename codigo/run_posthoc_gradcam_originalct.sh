#!/bin/bash
#SBATCH -J gradcamct
#SBATCH -p dios
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err
#SBATCH -c 4
#SBATCH --mem=24000

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta
CUDA_VISIBLE_DEVICES= /mnt/homeGPU/mcribilles/conda_envs/vc/bin/python posthoc_gradcam_multilabel.py
