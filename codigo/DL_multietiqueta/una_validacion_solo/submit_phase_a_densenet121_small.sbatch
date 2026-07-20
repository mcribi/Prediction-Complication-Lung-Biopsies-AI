#!/bin/bash
#SBATCH -J phAsDen
#SBATCH -p dgx
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err

set -euo pipefail

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python phase_a_small_medium_launcher.py --model densenet121 --size-group small
