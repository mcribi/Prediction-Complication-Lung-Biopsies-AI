#!/bin/bash
#SBATCH --job-name=tfm_subgroup_discovery
#SBATCH --output=logs/subgroup_discovery_%j.out
#SBATCH --error=logs/subgroup_discovery_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python3 run_subgroup_discovery.py \
  --n-jobs "${SLURM_CPUS_PER_TASK:-16}" \
  "$@"
