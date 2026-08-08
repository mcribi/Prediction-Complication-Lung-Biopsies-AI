# Subgroup Discovery

Análisis reproducible de subgrupos para la cohorte clínica de 210 pacientes del TFM.

## Archivos

- `run_subgroup_discovery.py`: implementación y CLI principal.
- `launch_subgroup_discovery.sbatch`: lanzamiento completo mediante SLURM en CPU.
- `tests/test_run_subgroup_discovery.py`: pruebas unitarias.
- `preliminar/`: código y resultados históricos, conservados sin cambios.
- `results/`: resultados de cada ejecución.
- `logs/`: salida y errores de SLURM.

## Dependencias

El script usa únicamente paquetes ya disponibles en `ngpu`:

- Python 3.8
- pandas 2.0.3
- NumPy 1.23.5
- SciPy 1.10.1
- scikit-learn 1.3.2
- joblib 1.4.2

No necesita GPU, `pysubgroup`, `pytest` ni `iterative-stratification`.

## Diseño implementado

Objetivos principales independientes:

- `Neumotórax`
- `Hemorragia`
- `Sin_complicación`

Conjuntos de variables:

- `clinical_full`: las 13 variables clínicas compactas en 210 pacientes.
- `clinical_matched`: las mismas variables clínicas en los 201 pacientes con geometría completa.
- `clinical_geometry`: la misma cohorte de 201 pacientes con las 13 variables clínicas y tres medidas geométricas.

Fuentes separadas por función:

- predictores clínicos: `clinical_data_multimodal.csv`;
- etiquetas objetivo: `clinical_data_limpios_SD.csv`;
- geometría del nódulo: `nodule_geometry_210.csv`.

Las 13 variables clínicas son `Edad`, `Sexo_binaria`, `Tabac_any`, `CardioRisk`, `Enfisema_any`, `Dislipemia_any`, `Hipertensión_pulmonar`, `DM_any`, `Obesidad`, `Fibrosis_any`, `AOS`, `Sin_factor_de_riesgo` y `Sin_patología_pulmonar`. El identificador se usa únicamente para emparejar tablas y no forma parte de las reglas.

La comparación entre `clinical_matched` y `clinical_geometry` utiliza exactamente los mismos pacientes.

Configuración principal:

- búsqueda en haz;
- WRAcc como función principal de calidad;
- longitud máxima de tres condiciones;
- cobertura mínima del 10 %;
- cinco reglas finales diversas por objetivo;
- eliminación de reglas con Jaccard de cobertura superior a 0,80;
- validación multietiqueta de 5 folds y 5 repeticiones;
- 1000 permutaciones de la búsqueda completa;
- sensibilidad para coberturas de 7,5 %, 10 % y 15 %, y profundidades 1, 2 y 3.

Los umbrales numéricos se calculan dentro del conjunto de entrenamiento. Los resultados no incluyen identificadores de pacientes.

## Pruebas

```bash
cd /mnt/homeGPU/mcribilles/tfm/codigo/subgroup_discovery
python3 -m unittest discover -s tests -v
```

## Ejecución rápida

Valida el pipeline con tres folds, una repetición y cinco permutaciones:

```bash
python3 run_subgroup_discovery.py --smoke
```

Los resultados rápidos son únicamente técnicos y no deben interpretarse como resultados finales.

## Experimento final con SLURM

```bash
cd /mnt/homeGPU/mcribilles/tfm/codigo/subgroup_discovery
sbatch launch_subgroup_discovery.sbatch
```

El comando devuelve un identificador de trabajo. Los logs se guardan en:

```text
logs/subgroup_discovery_<JOB_ID>.out
logs/subgroup_discovery_<JOB_ID>.err
```

## Ejecución directa

```bash
python3 run_subgroup_discovery.py --n-jobs 16
```

Las rutas anteriores son los valores predeterminados. Para utilizar copias alternativas de las mismas tres fuentes:

```bash
python3 run_subgroup_discovery.py \
  --clinical-csv /ruta/clinical_data_multimodal.csv \
  --targets-csv /ruta/clinical_data_limpios_SD.csv \
  --geometry-csv /ruta/nodule_geometry_210.csv
```

## Opciones útiles

Solo variables clínicas en los 210 pacientes:

```bash
python3 run_subgroup_discovery.py \
  --feature-sets clinical_full \
  --n-jobs 16
```

Añadir la etiqueta binaria histórica:

```bash
python3 run_subgroup_discovery.py \
  --include-binary-target \
  --n-jobs 16
```

Omitir permutaciones durante una prueba intermedia:

```bash
python3 run_subgroup_discovery.py \
  --skip-permutations \
  --n-repeats 1
```

Incorporar posteriormente objetivos de error calculados con predicciones out-of-fold:

```bash
python3 run_subgroup_discovery.py \
  --extra-target-csv /ruta/errores_oof.csv \
  --extra-targets error_neumotorax,error_hemorragia
```

El CSV adicional debe contener `patient_id` y los objetivos binarios indicados. Las predicciones deben ser estrictamente out-of-fold.

## Resultados

Cada ejecución crea una carpeta bajo `results/` con:

- `feature_audit.csv`: variables conservadas y descartadas.
- `rules_full.csv`: reglas descubiertas en toda la cohorte, de carácter exploratorio.
- `cv_rules.csv`: evaluación de cada regla en entrenamiento y test externo.
- `stability_summary.csv`: frecuencia de selección y comportamiento fuera de muestra.
- `permutation_summary.csv`: valor p ajustado por la búsqueda mediante permutaciones.
- `sensitivity_rules.csv`: sensibilidad a cobertura mínima y profundidad.
- `run_metadata.json`: rutas, hashes, versiones, semilla y configuración.
- `run.log`: registro completo de ejecución.

Las reglas describen asociaciones locales. No deben interpretarse como relaciones causales ni como evidencia clínica confirmatoria sin validación externa.
