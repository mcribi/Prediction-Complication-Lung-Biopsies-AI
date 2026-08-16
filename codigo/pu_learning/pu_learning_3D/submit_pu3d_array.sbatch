#!/bin/bash
#SBATCH --job-name=pu3d
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/pu_learning_3D/slurm_outputs/%A_%a.out
#SBATCH --error=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/pu_learning_3D/slurm_outputs/%A_%a.err

set -euo pipefail

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/pu_learning_3D
LABELS="$ROOT/manifest/labels_205.csv"
FOLDS="$ROOT/manifest/outer_folds.csv"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"

target_index=$((SLURM_ARRAY_TASK_ID / 5))
fold=$((SLURM_ARRAY_TASK_ID % 5 + 1))
if [[ "$target_index" -eq 0 ]]; then
    target=Hemorragia
else
    target=Neumotórax
fi

mkdir -p "$ROOT/slurm_outputs" "$RUN_DIR"
cd "$ROOT"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname) target=$target fold=$fold run_dir=$RUN_DIR"
/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python run_pu3d_fold.py \
    --target "$target" \
    --fold "$fold" \
    --labels-csv "$LABELS" \
    --folds-csv "$FOLDS" \
    --output-dir "$RUN_DIR" \
    --max-epochs "$MAX_EPOCHS"
