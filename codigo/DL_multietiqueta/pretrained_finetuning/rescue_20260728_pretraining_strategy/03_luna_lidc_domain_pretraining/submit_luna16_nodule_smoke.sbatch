#!/bin/bash
#SBATCH -J luna16Smoke
#SBATCH -p dgx,dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/03_luna_lidc_domain_pretraining

MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-64}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-32}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
RUN_TAG=${RUN_TAG:-slurm_gpu_smoke}

python train_luna16_nodule_smoke.py \
  --device cuda \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-val-samples "$MAX_VAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --run-tag "$RUN_TAG"
