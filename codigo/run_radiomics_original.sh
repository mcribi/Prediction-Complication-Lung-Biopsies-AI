#!/bin/bash
#SBATCH -J radiom
#SBATCH -p dgx
#SBATCH --cpus-per-task=8
#SBATCH --array=0-333%8             
#SBATCH -o /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.out
#SBATCH -e /mnt/homeGPU/mcribilles/tfm/slurm_outputs/%A_%a.err

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/mcribilles/conda_envs/vc

export CONFIG_FILE=./config/valid_configurations.HUGE.json
python ./radiomica/original_tfg/run_radiomic_experiment.py ${SLURM_ARRAY_TASK_ID}
