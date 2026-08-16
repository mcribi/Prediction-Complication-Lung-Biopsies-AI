#!/bin/bash
#SBATCH --job-name=pu3dAgg
#SBATCH --partition=dgx2,dgx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/pu_learning_3D/slurm_outputs/%j.out
#SBATCH --error=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/pu_learning_3D/slurm_outputs/%j.err

set -euo pipefail
ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/pu_learning_3D
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
cd "$ROOT"
/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python aggregate_pu3d.py --run-dir "$RUN_DIR"
