# Experimento multietiqueta multimodal

## Diseño

Se ejecutan 15 entrenamientos emparejados:

- 5 folds con variables clínicas;
- 5 folds con volúmenes 3D;
- 5 folds multimodales.

Las tres modalidades reutilizan exactamente los `train_inner`, `val_inner` y `test_outer` del barrido 3D seleccionado. El mejor epoch se elige por F1 micro en `val_inner`; `test_outer` se evalúa una única vez tras restaurar ese checkpoint.

La cabeza final recibe siempre una concatenación de 256 componentes de imagen y 64 componentes clínicos. En los controles unimodales, el bloque ausente se sustituye por ceros. Así se mantiene la misma cabeza de fusión en las tres modalidades.

La edad se estandariza de forma independiente en cada fold. La media y desviación se estiman exclusivamente con `train_inner` y se guardan en `clinical_transform.json`.

## Selección retrospectiva

La configuración 3D se ha seleccionado maximizando F1 micro OOF en `test_outer` entre configuraciones completas. Por ello, el experimento es exploratorio y sus resultados externos no constituyen una estimación confirmatoria no sesgada.

Configuración seleccionada:

- arquitectura: SE-ResNet-50;
- preprocesamiento: `resize_cube128_multiwindowing_separadas`;
- entrada: `ct_lung_nodule_vessels`;
- canales efectivos: 6;
- lote: 8;
- F1 micro OOF de origen: 0,5596;
- cohorte: 204 pacientes.

## Archivos

- `prepare_clinical_multimodal.py`: prepara las 13 variables clínicas.
- `selected_configuration.json`: registra la selección y el artefacto 3D de origen.
- `multimodal_common.py`: transformación clínica, dataset alineado y modelo de fusión.
- `run_multimodal_experiments.py`: entrenamiento, validación, test y agregación OOF.
- `tests/test_multimodal_common.py`: pruebas unitarias sin dependencias adicionales.
- `submit_smoke_test.sbatch`: prueba de integración en GPU.
- `submit_multimodal_15.sbatch`: ejecución secuencial y reanudable de los 15 folds.

## Ejecución

```bash
sbatch submit_smoke_test.sbatch
sbatch submit_multimodal_15.sbatch
```

El lanzador completo acepta un nombre de ejecución opcional:

```bash
sbatch submit_multimodal_15.sbatch nombre_ejecucion
```

Los resultados se guardan en `runs/<nombre_ejecucion>/`. La presencia de `complete.json` dentro de un fold permite omitirlo al reanudar. La agregación final solo se genera cuando existen los cinco folds de las tres modalidades y cada modalidad contiene exactamente una predicción OOF por paciente.
