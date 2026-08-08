#!/bin/bash
#SBATCH -J mm20dl
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%j.err

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta

source /opt/anaconda/etc/profile.d/conda.sh
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc

python run_top20_image_configs_multimodal_only.py \
  --limit 20 \
  --submit \
  --partition dgx2,dgx
