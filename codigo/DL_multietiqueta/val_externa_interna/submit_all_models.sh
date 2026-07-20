#!/bin/bash
set -euo pipefail

cd /mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/val_externa_interna

sbatch submit_nested_cv_model.sbatch resnet18 focused resnet18_focused
sbatch submit_nested_cv_model.sbatch resnet10 focused resnet10_focused
sbatch submit_nested_cv_model.sbatch resnet34 focused resnet34_focused
sbatch submit_nested_cv_model.sbatch seresnet50 focused seresnet50_focused
sbatch submit_nested_cv_model.sbatch densenet121 densenet_light densenet121_light
