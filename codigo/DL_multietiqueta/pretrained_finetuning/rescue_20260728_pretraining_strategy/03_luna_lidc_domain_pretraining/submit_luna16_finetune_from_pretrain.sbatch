#!/bin/bash
#SBATCH -J luna16FT
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

PRETRAINING=${PRETRAINING:-luna16_resnet34}
RUN_TAG=${RUN_TAG:-luna16_domain_pretrain_transfer}
BATCH_SIZE_OVERRIDE=${BATCH_SIZE_OVERRIDE:-}
LUNA_CHECKPOINT=${LUNA_CHECKPOINT:-}
CONFIG_INDEX=${CONFIG_INDEX:-0}

ARGS=(--pretraining "$PRETRAINING" --run-tag "$RUN_TAG" --config-index "$CONFIG_INDEX")
if [ -n "$BATCH_SIZE_OVERRIDE" ]; then
  ARGS+=(--batch-size-override "$BATCH_SIZE_OVERRIDE")
fi
if [ -n "$LUNA_CHECKPOINT" ]; then
  ARGS+=(--luna-checkpoint "$LUNA_CHECKPOINT")
fi

python pretrained_luna16_ct_simple.py "${ARGS[@]}"
