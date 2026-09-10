# PU learning

Esta sección estudia la posible incertidumbre de las etiquetas cuando algunos pacientes registrados sin una complicación podrían corresponder a positivos no identificados.

## Experimentos principales

### PU learning tabular

Se comparan modelos entrenados con variables clínicas y geométricas, características radiómicas y la combinación de ambas representaciones. Las formulaciones evaluadas son:

- supervisado convencional;
- supervisado con etiquetado negativo ingenuo;
- corrección de Elkan-Noto;
- Bagging PU.

La comparación utiliza un protocolo común de selección interna de modelos y umbrales para hemorragia y neumotórax.

### PU learning con imagen tridimensional

La extensión a imagen compara el entrenamiento supervisado, la corrección de Elkan-Noto y nnPU mediante una red tridimensional aplicada a los volúmenes y máscaras anatómicas.

También se incluye una simulación controlada de positivos ocultos para estudiar la robustez de los métodos cuando se modifica artificialmente una parte de las etiquetas de entrenamiento. Esta simulación no demuestra la existencia de complicaciones reales no registradas.

## Organización

- `baseline_supervisado/`: experimento tabular y comparación de formulaciones.
- `pu_learning_3D/`: extensión a volúmenes tridimensionales.
- `hidden_positive_simulation_20260901/`: simulación controlada de positivos ocultos.

La documentación técnica de ejecución se encuentra en los README internos de cada experimento.