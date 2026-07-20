#!/bin/bash
#SBATCH -J radiomica
#SBATCH -p dios
#SBATCH -w hera
#SBATCH --gres=gpu:0
#SBATCH -c 8
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A.err

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc

# python radiomica/adding_tabular_radiomic.py
# python ./aa_clasico/tabular_ml_grid.py \
#   --csv ../clinical_data/clinico_radiomica_con_tamano_y_profundidades.csv \
#   --target Complicacion_binaria \
#   --preprocessed_dir /mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/resize_medium/nifti \
#   --seed 42 \
#   --folds 5 \
#   --out_csv ./aa_clasico/resultados_ml_grid.csv

# python ./aa_clasico/tabular_ml_grid.py

python ./radiomica/original_tfg/make_configs_from_existing.py \
  --cv_root ./datasets \
  --recursive \
  --out_json config/valid_configurations.HUGE.json