#!/bin/bash
#SBATCH -J segmvessels
#SBATCH -p dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc

# python ./fisura_incompleta_pulmonar/encontrar_fisuras.py \
#   --lobes_dir ../volumenes_preprocesados/lobes_masks_from_dicom \
#   --out_csv ./fisura_incompleta_pulmonar/fisuras_flags.csv

# python fisura_incompleta_pulmonar/detectar_fisuras_incompletas_CIP.py
python ../segmentation/segmentar_vessels.py 