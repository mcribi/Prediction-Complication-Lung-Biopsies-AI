#!/bin/bash
#SBATCH --job-name=tfm25d
#SBATCH --output=slurm_outputs/%A_%a.out
#SBATCH --error=slurm_outputs/%A_%a.err
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-9%2

set -euo pipefail

MODULE_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be defined}"
cd "$MODULE_DIR"
mkdir -p slurm_outputs

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc

: "${RUN_DIR:?RUN_DIR must be defined}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 5 )); then
    REPRESENTATION="axial5"
    FOLD=$((TASK_ID + 1))
else
    REPRESENTATION="triplanar"
    FOLD=$((TASK_ID - 4))
fi

python run_25d_fold.py \
    --representation "$REPRESENTATION" \
    --fold "$FOLD" \
    --run-dir "$RUN_DIR" \
    --max-epochs "$MAX_EPOCHS"
