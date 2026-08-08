#!/bin/bash
#SBATCH -J luna16Pretr
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/03_luna_lidc_domain_pretraining

ARCHITECTURE=${ARCHITECTURE:-resnet34}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-2048}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-512}
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-5}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-1e-4}
RUN_TAG=${RUN_TAG:-bounded_real}

python train_luna16_nodule_pretrain.py \
  --device cuda \
  --architecture "$ARCHITECTURE" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-val-samples "$MAX_VAL_SAMPLES" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --run-tag "$RUN_TAG"
