#!/bin/bash
#SBATCH --job-name=pu_hidden_agg
#SBATCH --partition=muylarga
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python}"
export PYTHONPATH="$CODE_ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"$PYTHON" \
  "$CODE_ROOT/aggregate_hidden_positive.py" \
  --run-root "$RUN_ROOT"
