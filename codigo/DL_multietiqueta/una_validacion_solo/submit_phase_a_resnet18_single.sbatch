#!/bin/bash
#SBATCH -J phaAres
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err

set -euo pipefail
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta
export PYTHONUNBUFFERED=1
python phase_a_resnet18_serial.py
