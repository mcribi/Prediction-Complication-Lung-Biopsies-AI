# Val externa e interna para DL multietiqueta

Objetivo:
- corregir el sesgo del pipeline anterior separando seleccion de epoch y evaluacion final
- mantener todos los preprocesados, incluyendo `small` y `medium`
- lanzar un job por modelo para paralelizar facilmente

## Esquema de validacion

Para cada configuracion:
1. CV externa estratificada en 5 folds sobre `labelset_key`
2. En cada fold externo:
   - `test_outer`: 20% aprox del total
   - `train_outer`: 80% aprox del total
3. Dentro de `train_outer`:
   - `train_inner`: 80% de `train_outer`
   - `val_inner`: 20% de `train_outer`
4. `val_inner` se usa solo para:
   - early stopping
   - `best_epoch`
   - `best_model.pt`
5. `test_outer` se usa solo para:
   - metricas finales del fold
   - `test_predictions_best_epoch.csv`
6. Las metricas OOF se construyen solo con `test_outer` de los 5 folds

## Scripts

- `dl_nested_cv_common.py`: runner comun con split externo/interno
- `run_nested_cv_model.py`: entrada generica por modelo
- `run_resnet18.py`
- `run_resnet10.py`
- `run_resnet34.py`
- `run_seresnet50.py`
- `run_densenet121.py`
- `submit_nested_cv_model.sbatch`: plantilla Slurm
- `submit_all_models.sh`: envia 5 jobs, uno por modelo
- `posthoc_gradcam_nestedcv.py`: Grad-CAM sobre test externo
- `submit_posthoc_gradcam_nestedcv.sbatch`: job Slurm para Grad-CAM posthoc

## Politica actual para acotar el barrido

Se usan todos los preprocesados detectados en disco, pero se acota por:

- 1 batch size por familia espacial, guiado por el historico:
  - cube64 -> `bs=16`
  - cube128 -> `bs=8`
  - small -> `bs=4`
  - medium -> `bs=1` con `grad_accum_steps=4`
- perfiles de input:
  - `focused`: `nodule_only_masked_ct`, `ct_lung_nodule`, `ct_lung_nodule_vessels`
  - `densenet_light`: `nodule_only_masked_ct`, `ct_lung_nodule`

## Justificacion del acotado

Tendencias utiles del historico anterior, solo como guia de exploracion:

- modelos:
  - `resnet34` fue el mejor
  - `resnet18` se mantuvo fuerte y estable
  - `resnet10` y `seresnet50` quedaron en zona intermedia
  - `densenet121` quedo claramente peor y ademas es costoso
- inputs:
  - mejor rendimiento historico en `nodule_only_masked_ct`
  - despues `ct_lung_nodule`
  - despues `ct_lung_nodule_vessels`
  - `lung_only_masked_ct` y `nodule_vessels` rindieron peor
- tamanos:
  - `cube128` y `small` dieron los mejores techos
  - `medium` solo debe mantenerse con `bs=1` y acumulacion

## Comandos utiles

Un modelo:

```bash
python run_nested_cv_model.py --model resnet34 --input-profile focused --run-tag prueba
```

Cinco jobs:

```bash
bash submit_all_models.sh
```

## Artefactos por fold

Cada fold debe generar:

- `history.csv`
- `best_model.pt`
- `split_assignments.csv`
- `val_predictions_best_epoch.csv`
- `test_predictions_best_epoch.csv`

Cada configuracion debe generar:

- `fold_summary.csv`
- `oof_predictions.csv`
- `oof_metrics.json`

## Grad-CAM del test externo

El script `posthoc_gradcam_nestedcv.py` toma las mejores configuraciones por `oof_f1_micro` y, para cada una, selecciona desde `oof_predictions.csv` del test externo:

- `Hemorragia`: 1 correcto (`tp`) + 1 incorrecto (`fn`)
- `Neumotórax`: 1 correcto (`tp`) + 1 incorrecto (`fn`)
- `Sin_complicacion`: 1 correcto (`tp`) + 1 incorrecto (`fn`)

Total: 6 casos por configuracion.

Alineacion visual:

- guarda la CAM en la rejilla preprocesada del modelo
- y tambien la reproyecta al CT fuente con la transposicion inversa y `inverse-resize`
- no aplica rotaciones manuales ad hoc; la correspondencia se hace siguiendo el orden de ejes del preprocessing

## Nota metodologica

El historico antiguo no debe usarse para reportar metricas finales. Solo se usa para elegir una direccion de barrido mas razonable.
