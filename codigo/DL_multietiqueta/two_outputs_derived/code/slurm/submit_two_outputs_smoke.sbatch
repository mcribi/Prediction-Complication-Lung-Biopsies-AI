#!/bin/bash
#SBATCH -J dl2smoke
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
ORIGINAL_MM=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/image_two_outputs:$CODE_ROOT/multimodal_two_outputs:$CODE_ROOT/pretrained_two_outputs:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna:/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/una_validacion_solo:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$CODE_ROOT"
python -m unittest discover -s tests -v
python -m py_compile \
  two_output_metrics.py \
  collect_two_output_results.py \
  image_two_outputs/dl_nested_cv_two_outputs.py \
  image_two_outputs/run_nested_cv_model.py \
  multimodal_two_outputs/multimodal_common.py \
  multimodal_two_outputs/run_multimodal_experiments.py \
  pretrained_two_outputs/pretrained_freeze_two_outputs.py \
  pretrained_two_outputs/pretrained_nested_cv_pilots.py \
  tests/smoke_project_setup.py

python tests/smoke_project_setup.py

TOP1=$(python -c 'from pathlib import Path; files=sorted(Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta/manifests_top20_multimodal_only").glob("top01_*.json")); assert len(files)==1, files; print(files[0])')
python multimodal_two_outputs/run_multimodal_experiments.py \
  --manifest "$TOP1" \
  --smoke-test \
  --num-workers 0

echo TWO_OUTPUT_SMOKE_OK
