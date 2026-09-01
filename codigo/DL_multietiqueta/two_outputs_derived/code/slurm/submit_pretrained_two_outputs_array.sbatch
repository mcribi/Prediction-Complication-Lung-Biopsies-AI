#!/bin/bash
#SBATCH -J dl2pre
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-1%1
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived
CODE_ROOT="$ROOT/code"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/pretrained_two_outputs:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/una_validacion_solo:${PYTHONPATH:-}"
cd "$CODE_ROOT/pretrained_two_outputs"

PRETRAININGS=(medicalnet genesis_chest_ct)
PRETRAINING="${PRETRAININGS[$SLURM_ARRAY_TASK_ID]}"
python pretrained_nested_cv_pilots.py \
  --pretraining "$PRETRAINING" \
  --run-tag two_outputs_thresholds
