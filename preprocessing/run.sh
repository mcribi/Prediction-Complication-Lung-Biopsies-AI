#!/bin/bash
#SBATCH -J pre_mask
#SBATCH -p dios
#SBATCH -w hera
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
# python ./preprocess_and_save_all.py

python comprobar_preprocesamientos.py \
  --root /mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/ \
  --formats npy,nifti \
  --output preproc_qc_report.csv
