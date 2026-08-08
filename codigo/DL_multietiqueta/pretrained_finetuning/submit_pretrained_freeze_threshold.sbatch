#!/bin/bash
#SBATCH -J preFTth
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -eo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u
cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python pretrained_freeze_threshold_pilots.py \
  --pretraining "${PRETRAINING}" \
  --run-tag "phase2_freeze_unfreeze_threshold_${PRETRAINING}"
