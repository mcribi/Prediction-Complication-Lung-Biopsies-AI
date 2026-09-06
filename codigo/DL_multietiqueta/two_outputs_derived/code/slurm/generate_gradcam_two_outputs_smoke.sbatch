#!/bin/bash
#SBATCH --job-name=gradcam2smk
#SBATCH -p dios
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH --error=/mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail
PYTHON=/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python
CODE_DIR=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived/code/image_two_outputs
OUTPUT_DIR=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived/image_only/gradcam_posthoc/smoke_${SLURM_JOB_ID}

cd "$CODE_DIR"
$PYTHON -m unittest discover -s tests -p 'test_gradcam_two_outputs_image.py' -v
$PYTHON gradcam_two_outputs_image.py \
  --output-dir "$OUTPUT_DIR" \
  --top-k-configs 1 \
  --cases-per-state 1 \
  --max-cases-per-config 1
