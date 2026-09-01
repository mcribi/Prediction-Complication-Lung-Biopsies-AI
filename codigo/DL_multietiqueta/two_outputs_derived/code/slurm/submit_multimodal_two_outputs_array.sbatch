#!/bin/bash
#SBATCH -J dl2mm
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --array=1-20%2
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.err

set -euo pipefail
set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived
CODE_ROOT="$ROOT/code"
ORIGINAL_MM=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/multimodal_two_outputs:${PYTHONPATH:-}"
cd "$CODE_ROOT/multimodal_two_outputs"

RANK=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
MANIFEST=$(python -c 'from pathlib import Path; import sys; files=sorted(Path(sys.argv[1]).glob(f"top{sys.argv[2]}_*.json")); assert len(files)==1, files; print(files[0])' "$ORIGINAL_MM/manifests_top20_multimodal_only" "$RANK")
RUN_NAME="top${RANK}_two_outputs_multimodal"

python run_multimodal_experiments.py \
  --manifest "$MANIFEST" \
  --run-name "$RUN_NAME" \
  --modes multimodal \
  --folds 1 2 3 4 5 \
  --max-epochs 50 \
  --patience 10 \
  --num-workers 4
