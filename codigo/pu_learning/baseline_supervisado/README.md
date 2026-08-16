# Baseline supervisado inicial para PU learning

Scripts iniciales para comparar tres bloques de variables con la misma validacion cruzada estratificada usada en radiomica:

1. `clinical_geometry`: variables clinicas SD + geometria.
2. `radiomics`: radiomica basica con mascaras `lung+nodule+vessels`.
3. `clinical_geometry_radiomics`: combinacion de las dos anteriores.

La particion se construye una sola vez sobre la interseccion de pacientes comun a los tres CSV y se reutiliza en todas las configuraciones. Por defecto se usa `StratifiedKFold` de 5 folds sobre la firma multietiqueta de `Hemorragia`, `Neumotorax` y `Sin_complicacion`, con semilla 42.

## Preparar grid

```bash
cd /mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado
/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python build_supervised_baseline_grid.py
```

Esto genera:

- `baseline_grid/valid_configurations_supervised_baseline.json`
- `baseline_grid/shared_cohort_patient_ids.csv`
- `baseline_grid/grid_summary.json`

## Smoke test

```bash
cd /mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado
/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python run_supervised_baseline.py 0 \
  --config_file baseline_grid/valid_configurations_supervised_baseline.json \
  --output_root results/smoke \
  --dry_run
```

## Lanzar en SLURM

```bash
cd /mnt/homeGPU/mcribilles/tfm/codigo/pu_learning/baseline_supervisado
sbatch run_supervised_baseline_array.sbatch
```

## Reutilizar una particion exacta existente

Si se quiere forzar exactamente un `split_assignments.csv` concreto de radiomica, ejecutar una configuracion con:

```bash
/mnt/homeGPU/mcribilles/conda_envs/vc/bin/python run_supervised_baseline.py 0 \
  --config_file baseline_grid/valid_configurations_supervised_baseline.json \
  --output_root results/with_reference_folds \
  --folds_csv /ruta/al/split_assignments.csv \
  --save_predictions
```

Nota: si se reutiliza un `split_assignments.csv` de una cohorte distinta, el script conserva solo los pacientes presentes en el bloque actual. Para comparacion estricta entre los tres bloques, es preferible usar la particion comun generada por `build_supervised_baseline_grid.py`.
