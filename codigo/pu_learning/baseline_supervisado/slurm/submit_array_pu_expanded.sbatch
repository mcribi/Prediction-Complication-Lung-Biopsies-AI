#!/bin/bash
#SBATCH --job-name=pu_grid_expanded
#SBATCH --output=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado/slurm_outputs/pu_grid_%A_%a.out
#SBATCH --error=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado/slurm_outputs/pu_grid_%A_%a.err
#SBATCH --array=0-3671%80
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00

set -euo pipefail

cd /mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado

PYTHON=/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python
MANIFEST=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado/manifests/expanded_grid_20260811_1205/expanded_grid_manifest.csv

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

$PYTHON scripts/run_one_experiment.py \
  --manifest "$MANIFEST" \
  --task-id "$SLURM_ARRAY_TASK_ID"
