#!/bin/bash
#SBATCH -J dlNest
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail

MODEL="${1:?model required}"
INPUT_PROFILE="${2:-}"
RUN_TAG="${3:-}"

set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna

CMD=(python run_nested_cv_model.py --model "$MODEL")
if [[ -n "$INPUT_PROFILE" ]]; then
  CMD+=(--input-profile "$INPUT_PROFILE")
fi
if [[ -n "$RUN_TAG" ]]; then
  CMD+=(--run-tag "$RUN_TAG")
fi

echo "MODEL=$MODEL"
echo "INPUT_PROFILE=${INPUT_PROFILE:-default}"
echo "RUN_TAG=${RUN_TAG:-none}"
printf 'CMD='
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
