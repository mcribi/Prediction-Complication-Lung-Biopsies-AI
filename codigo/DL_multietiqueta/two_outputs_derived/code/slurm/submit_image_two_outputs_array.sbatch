#!/bin/bash
#SBATCH -J dl2img
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-19%2
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived
CODE_ROOT="$ROOT/code"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/image_two_outputs:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/una_validacion_solo:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$CODE_ROOT/image_two_outputs"

MODELS=(resnet10 resnet18 resnet34 seresnet50 densenet121)
PREPROCESSING_GROUPS=(cube64 cube128 small medium)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
GROUP_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
MODEL="${MODELS[$MODEL_INDEX]}"
GROUP="${PREPROCESSING_GROUPS[$GROUP_INDEX]}"

case "$GROUP" in
  cube64) PREFIX=resize_cube64 ;;
  cube128) PREFIX=resize_cube128 ;;
  small) PREFIX=resize_small ;;
  medium) PREFIX=resize_medium ;;
  *) exit 2 ;;
esac

PREPROCESSINGS=(
  "$PREFIX"
  "${PREFIX}_hu_m300_1400"
  "${PREFIX}_hu_m600_1500"
  "${PREFIX}_multiwindowing_separadas"
)

if [[ "$MODEL" == "densenet121" ]]; then
  PROFILE=densenet_light
else
  PROFILE=focused
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN_OK task=%s model=%s group=%s profile=%s preprocessings=%s\n' \
    "$SLURM_ARRAY_TASK_ID" "$MODEL" "$GROUP" "$PROFILE" "${PREPROCESSINGS[*]}"
  exit 0
fi

python run_nested_cv_model.py \
  --model "$MODEL" \
  --input-profile "$PROFILE" \
  --run-tag "two_outputs_${GROUP}" \
  --output-root "$ROOT/image_only/runs" \
  --preprocessing-names "${PREPROCESSINGS[@]}"
