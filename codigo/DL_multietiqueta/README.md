# Deep learning multietiqueta

Esta sección contiene los modelos que procesan directamente los volúmenes tridimensionales de tomografía computarizada para predecir hemorragia y neumotórax. La ausencia de complicaciones se deriva cuando ambas salidas son negativas.

## Experimentos principales

- Modelos tridimensionales basados únicamente en imagen, comparando arquitecturas y representaciones preprocesadas de los volúmenes.
- Modelos multimodales que combinan una rama de imagen tridimensional con variables clínicas.
- Transferencia de aprendizaje y ajuste fino a partir de modelos preentrenados en imagen médica.
- Análisis Grad-CAM de los modelos seleccionados para estudiar las regiones que influyen en sus predicciones.

La selección de checkpoints y umbrales se realiza con pacientes de validación interna. La evaluación final de cada partición utiliza pacientes que no han intervenido en el entrenamiento ni en dicha selección.

## Organización

- `two_outputs_derived/code/image_two_outputs/`: modelos tridimensionales basados en imagen.
- `two_outputs_derived/code/multimodal_two_outputs/`: modelos multimodales.
- `two_outputs_derived/code/pretrained_two_outputs/`: transferencia de aprendizaje y ajuste fino.
- `two_outputs_derived/code/slurm/`: lanzadores de los experimentos principales.
- `two_outputs_derived/code/README.md`: documentación técnica y artefactos generados.

Los volúmenes, checkpoints, predicciones y mapas Grad-CAM por paciente no se incluyen en el repositorio.