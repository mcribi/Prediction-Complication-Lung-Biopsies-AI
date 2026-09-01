#!/bin/bash
#SBATCH --job-name=tfm25dAgg
#SBATCH --output=slurm_outputs/%j.out
#SBATCH --error=slurm_outputs/%j.err
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00

set -euo pipefail
MODULE_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be defined}"
cd "$MODULE_DIR"
mkdir -p slurm_outputs
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u
: "${RUN_DIR:?RUN_DIR must be defined}"
python aggregate_25d.py --run-dir "$RUN_DIR"
