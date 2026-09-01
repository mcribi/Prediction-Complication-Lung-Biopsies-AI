#!/bin/bash
#SBATCH --job-name=sgradagg
#SBATCH --partition=muylarga
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${MANIFEST:?MANIFEST is required}"
CODE_DIR=/home/mcribilles/tfm/subgroup_radiomics_ml_20260831
PYTHON=/home/mcribilles/miniconda3/envs/pyradiomics_env/bin/python
export PYTHONUNBUFFERED=1
cd "$CODE_DIR"
"$PYTHON" aggregate_radiomics_subgroups.py --manifest "$MANIFEST" --output-dir "$RUN_ROOT/aggregate"
