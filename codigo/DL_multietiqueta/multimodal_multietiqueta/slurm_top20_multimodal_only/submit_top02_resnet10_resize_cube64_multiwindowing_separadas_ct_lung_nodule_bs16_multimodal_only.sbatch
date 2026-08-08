#!/bin/bash
#SBATCH -J mm_top20
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
cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta
python /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta/prepare_clinical_multimodal.py
PYTHONPATH=. python /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta/tests/test_multimodal_common.py -v
python /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta/run_multimodal_experiments.py   --manifest /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/multimodal_multietiqueta/manifests_top20_multimodal_only/top02_resnet10_resize_cube64_multiwindowing_separadas__ct_lung_nodule__bs16.json   --run-name top02_resnet10_resize_cube64_multiwindowing_separadas__ct_lung_nodule__bs16_multimodal_only   --modes multimodal   --max-epochs 50   --patience 10   --num-workers 4
