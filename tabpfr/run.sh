#!/bin/bash
#SBATCH -J lung3d
#SBATCH -p dios
#SBATCH --gres=gpu:0
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/tabpfn


python ./tipo_complicacion.py 