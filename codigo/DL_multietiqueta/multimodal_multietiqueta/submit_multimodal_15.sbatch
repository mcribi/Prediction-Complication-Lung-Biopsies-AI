#!/bin/bash
#SBATCH -J mm15
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

WORKDIR=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta
RUN_NAME="${1:-seresnet50_best_test_f1_15runs}"
cd "$WORKDIR"
python prepare_clinical_multimodal.py
PYTHONPATH=. python tests/test_multimodal_common.py -v
python run_multimodal_experiments.py --run-name "$RUN_NAME"
