#!/bin/bash
#SBATCH -J dl2freeze
#SBATCH -p dgx2,dgx
#SBATCH --array=0-9%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived
cd "$ROOT/code"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$ROOT/code:$ROOT/code/pretrained_two_outputs:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/una_validacion_solo:${PYTHONPATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) PRETRAINING=medicalnet; PROFILE=phase2; CONFIG_INDEX=0 ;;
  1) PRETRAINING=medicalnet; PROFILE=phase2; CONFIG_INDEX=1 ;;
  2) PRETRAINING=genesis_chest_ct; PROFILE=phase2; CONFIG_INDEX=0 ;;
  3) PRETRAINING=genesis_chest_ct; PROFILE=phase2; CONFIG_INDEX=1 ;;
  4) PRETRAINING=medicalnet; PROFILE=ct_simple; CONFIG_INDEX=0 ;;
  5) PRETRAINING=medicalnet; PROFILE=ct_simple; CONFIG_INDEX=1 ;;
  6) PRETRAINING=genesis_chest_ct; PROFILE=ct_simple; CONFIG_INDEX=0 ;;
  7) PRETRAINING=genesis_chest_ct; PROFILE=ct_simple; CONFIG_INDEX=1 ;;
  8) PRETRAINING=luna16_resnet34; PROFILE=luna16; CONFIG_INDEX=0 ;;
  9) PRETRAINING=luna16_resnet34; PROFILE=luna16; CONFIG_INDEX=1 ;;
  *) echo "Invalid task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

python pretrained_two_outputs/pretrained_freeze_two_outputs.py \
  --pretraining "$PRETRAINING" \
  --profile "$PROFILE" \
  --config-index "$CONFIG_INDEX" \
  --run-tag "two_outputs_${PRETRAINING}_${PROFILE}_cfg${CONFIG_INDEX}"
