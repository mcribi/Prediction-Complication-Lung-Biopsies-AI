#!/bin/bash
#SBATCH --job-name=pu_hide_tests
#SBATCH --partition=muylarga
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python}"
mkdir -p "$ROOT/slurm_outputs"
cd "$ROOT"
export PYTHONPATH="$ROOT"
"$PYTHON" -m unittest -v tests/test_hidden_positive_simulation.py
