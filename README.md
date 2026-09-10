# Técnicas avanzadas de Inteligencia Artificial para predicción y caracterización de complicaciones en biopsias pulmonares

> 🌐 **Web interactiva de resultados del TFM:** [Acceder a la web](https://mcribi.github.io/Prediction-Complication-Lung-Biopsies-AI/)

Repositorio de código del Trabajo de Fin de Máster dedicado al estudio de complicaciones asociadas a biopsias pulmonares guiadas por tomografía computarizada.

El proyecto combina información clínica, características geométricas, radiómica y volúmenes de tomografía computarizada. La formulación principal utiliza dos salidas independientes:

- Hemorragia.
- Neumotórax.

El estado `Sin_complicación` se deriva cuando ambas salidas son negativas y no se entrena como una salida adicional.

## Objetivos

- Abordar la predicción multimodal y multietiqueta de complicaciones mediante técnicas avanzadas de inteligencia artificial y análisis de datos.
- Analizar la relevancia de la información clínica, radiómica y de imagen médica para la capacidad predictiva de los modelos.
- Explorar metodologías adaptadas al desbalanceo de clases, la posible incertidumbre de las etiquetas y la identificación de subgrupos relevantes de pacientes.
- Diseñar y evaluar modelos predictivos explicables, considerando tanto su rendimiento como la utilidad de sus interpretaciones en un entorno clínico.
- Extraer conclusiones clínicas y metodológicas que permitan comprender mejor el problema y orientar futuras líneas de investigación.

## Estructura del repositorio

```text
.
├── codigo/
│   ├── eda/                         # Análisis exploratorio y preparación de datos
│   ├── radiomica_multietiqueta/     # Experimentos de radiómica
│   ├── DL_multietiqueta/            # Modelos de deep learning con dos salidas
│   ├── pu_learning/                  # Experimentos de PU learning
│   ├── subgroup_discovery/           # Descubrimiento y evaluación de subgrupos
│   ├── fisura_incompleta_pulmonar/   # Análisis de fisuras pulmonares
│   └── clasif_binaria/               # Código histórico de clasificación binaria
├── preprocessing/                    # Preprocesado de volúmenes
├── segmentation/                     # Segmentación de estructuras pulmonares
├── memoria_latex/                    # Fuentes LaTeX de la memoria
└── .gitignore                        # Exclusión de datos y artefactos sensibles
```

Los directorios de datos, segmentaciones, volúmenes, pesos, predicciones y resultados regenerables no se distribuyen en el repositorio.

## Metodología

El flujo general del proyecto comprende:

1. Limpieza, análisis exploratorio y preparación de los datos clínicos y de imagen.
2. Preprocesado y segmentación de los volúmenes médicos.
3. Extracción de características geométricas y radiómicas.
4. Entrenamiento y evaluación de modelos radiómicos y de deep learning, incluyendo enfoques basados únicamente en imagen y modelos multimodales.
5. Aplicación de PU learning para estudiar la posible incertidumbre de las etiquetas mediante representaciones tabulares y volúmenes tridimensionales.
6. Descubrimiento de subgrupos para identificar reglas interpretables y estudiar modelos específicos en grupos concretos de pacientes.
7. Análisis de explicabilidad mediante SHAP para los modelos radiómicos y Grad-CAM para los modelos tridimensionales.

Las particiones se realizan por paciente. La selección de modelos, umbrales e hiperparámetros se mantiene separada de la evaluación final para reducir el riesgo de fuga de información.

## Documentación de los experimentos

Cada línea experimental dispone de un README de entrada con sus experimentos principales:

- [Radiómica multietiqueta](codigo/radiomica_multietiqueta/README.md)
- [Deep learning multietiqueta](codigo/DL_multietiqueta/README.md)
- [PU learning](codigo/pu_learning/README.md)
- [Descubrimiento de subgrupos](codigo/subgroup_discovery/README.md)

Los detalles técnicos de ejecución se documentan junto a los scripts de cada experimento.

## Requisitos y ejecución

El repositorio contiene experimentos con requisitos distintos. Antes de ejecutar un experimento, debe consultarse su README específico y adaptar las rutas de entrada a la infraestructura local.

Los trabajos de cómputo intensivo incluyen scripts para SLURM. Se recomienda ejecutar primero las pruebas unitarias o pruebas técnicas disponibles en el módulo correspondiente antes de lanzar un experimento completo.

## Datos y privacidad

Los datos clínicos y las imágenes médicas no forman parte del repositorio. El archivo `.gitignore` excluye, entre otros elementos:

- datos clínicos e identificadores de pacientes;
- imágenes situadas en carpetas de pacientes;
- volúmenes médicos y segmentaciones derivadas;
- asignaciones de particiones y predicciones por paciente;
- checkpoints, pesos y resultados regenerables;
- credenciales y configuraciones locales.
