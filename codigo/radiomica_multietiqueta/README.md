# Radiómica multietiqueta

Esta sección reúne los experimentos basados en características cuantitativas extraídas de las imágenes médicas para la predicción de hemorragia y neumotórax.

## Experimentos principales

- Extracción de características radiómicas sobre imágenes originales y preprocesadas, utilizando distintas estructuras anatómicas segmentadas.
- Análisis exploratorio, control de redundancia y preprocesamiento de las características dentro del flujo de validación.
- Comparación de modelos clásicos con características radiómicas, variables clínicas y combinaciones de ambas fuentes.
- Evaluación de distintas representaciones de imagen y máscaras anatómicas bajo un protocolo común.
- Análisis explicativo de los modelos radiómicos seleccionados mediante SHAP.

## Organización

- `experiments/radiomica/`: construcción de configuraciones, extracción de características y ejecución de los experimentos multietiqueta.
- `shap_radiomics_final_20260903/`: análisis SHAP de los modelos radiómicos seleccionados y generación de figuras.

Los datos de entrada, resultados, predicciones y artefactos explicativos por paciente se mantienen fuera del control de versiones.