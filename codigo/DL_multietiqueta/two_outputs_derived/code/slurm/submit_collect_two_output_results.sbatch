#!/bin/bash
#SBATCH -J dl2collect
#SBATCH -p dgx2,dgx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived
CODE_ROOT="$ROOT/code"
TABLE_ROOT="$ROOT/tables"
MIN_MTIME_EPOCH=${MIN_MTIME_EPOCH:?MIN_MTIME_EPOCH must identify the current launch batch}
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
TABLE_ROOT="$TABLE_ROOT/launch_${MIN_MTIME_EPOCH}"
mkdir -p "$TABLE_ROOT"

python "$CODE_ROOT/collect_two_output_results.py" \
  --root "$ROOT" \
  --min-mtime-epoch "$MIN_MTIME_EPOCH" \
  --output "$TABLE_ROOT/master_two_output_results.csv"
