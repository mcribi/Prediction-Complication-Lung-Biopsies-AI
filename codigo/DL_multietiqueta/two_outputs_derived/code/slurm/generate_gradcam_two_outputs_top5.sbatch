#!/bin/bash
#SBATCH --job-name=gradcam2img
#SBATCH -p dios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH --error=/mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail
PYTHON=/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python
CODE_DIR=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived/code/image_two_outputs
OUTPUT_DIR=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived/image_only/gradcam_posthoc/top5_completed_${SLURM_JOB_ID}

cd "$CODE_DIR"
$PYTHON -m unittest discover -s tests -p 'test_gradcam_two_outputs_image.py' -v
$PYTHON gradcam_two_outputs_image.py \
  --output-dir "$OUTPUT_DIR" \
  --top-k-configs 5 \
  --cases-per-state 1
