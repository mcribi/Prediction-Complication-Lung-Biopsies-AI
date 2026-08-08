#!/bin/bash
#SBATCH -J dlPretr
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

set -euo pipefail

PRETRAINING="${1:?pretraining required: medicalnet or genesis_chest_ct}"
RUN_TAG="${2:-priority1_pilot}"

set +u
source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc
set -u

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/pretrained_finetuning

CMD=(python pretrained_nested_cv_pilots.py --pretraining "$PRETRAINING" --run-tag "$RUN_TAG")

echo "PRETRAINING=$PRETRAINING"
echo "RUN_TAG=$RUN_TAG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
printf 'CMD='
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
