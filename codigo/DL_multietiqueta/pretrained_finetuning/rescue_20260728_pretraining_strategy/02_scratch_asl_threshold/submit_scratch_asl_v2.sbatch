#!/bin/bash
#SBATCH -J aslV2
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
cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning/rescue_20260728_pretraining_strategy/02_scratch_asl_threshold
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
EXTRA_ARGS=()
if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--batch-size-override "${BATCH_SIZE_OVERRIDE}")
fi
if [[ -n "${CONFIG_INDEXES:-}" ]]; then
  IFS=',' read -ra IDX_ARR <<< "${CONFIG_INDEXES}"
  for idx in "${IDX_ARR[@]}"; do
    EXTRA_ARGS+=(--config-index "${idx}")
  done
fi
python scratch_asl_threshold_v2.py --pretraining scratch_asl --run-tag top_configs_scratch_asl_threshold "${EXTRA_ARGS[@]}"
