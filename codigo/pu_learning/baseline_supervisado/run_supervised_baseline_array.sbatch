#!/bin/bash
#SBATCH --job-name=pu_sup_base
#SBATCH --output=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado/slurm_outputs/%x_%A_%a.out
#SBATCH --error=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado/slurm_outputs/%x_%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-17%6

set -euo pipefail

PYTHON=/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python
BASE=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado
CONFIG_FILE=${CONFIG_FILE:-$BASE/baseline_grid/valid_configurations_supervised_baseline.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-$BASE/results/supervised_baseline_initial}
CONFIG_INDEX=${SLURM_ARRAY_TASK_ID:-0}

mkdir -p "$BASE/slurm_outputs" "$OUTPUT_ROOT"

$PYTHON "$BASE/run_supervised_baseline.py" "$CONFIG_INDEX" \
  --config_file "$CONFIG_FILE" \
  --output_root "$OUTPUT_ROOT" \
  --save_predictions
