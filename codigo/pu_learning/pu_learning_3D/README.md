# PU learning con volúmenes 3D

## Objetivo

Comparar aprendizaje supervisado, corrección Elkan-Noto y nnPU para Hemorragia y Neumotórax utilizando volúmenes 3D.

## Diseño

- Cohorte común: 205 pacientes.
- Folds externos: los mismos cinco folds del experimento PU tabular definitivo.
- Unidad de partición: paciente.
- Entrada: `resize_cube64`, canales `ct_lung`, `lung` y `nodule`.
- Arquitectura: ResNet-18 3D.
- Entrenamientos por objetivo y fold:
  - BCE supervisada sobre positivos observados frente a no etiquetados;
  - nnPU.
- Elkan-Noto reutiliza el modelo supervisado y corrige sus probabilidades mediante `p(s=1|x) / c`.
- Checkpoint: mayor average precision en `val_inner`.
- Umbral: maximiza G-mean, con MCC y F1 como desempates, en `val_inner`.
- `c`: media de `p(s=1|x)` en positivos observados de `val_inner`.
- Prior nnPU: prevalencia observada del entrenamiento externo dividida por `c`.
- `test_outer` solo se utiliza para evaluación final OOF.

## Ejecución

Cada tarea del array corresponde a un objetivo y fold externo. El array completo contiene diez tareas, con un máximo de cuatro GPU simultáneas.

```bash
RUN_DIR=/ruta/del/run sbatch --export=ALL,RUN_DIR="$RUN_DIR" submit_pu3d_array.sbatch
```

Tras completarse, se ejecuta `aggregate_pu3d.sbatch` con dependencia `afterok`.

## Salidas

Cada tarea guarda asignaciones de pacientes, historiales, checkpoints, probabilidades externas y métricas. El agregador verifica los cinco folds y los 205 pacientes OOF antes de calcular métricas agregadas.
