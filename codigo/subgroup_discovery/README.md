# Descubrimiento de subgrupos

Esta sección estudia perfiles concretos de pacientes mediante reglas interpretables. Su finalidad es describir asociaciones locales y explorar diferencias entre grupos, no establecer relaciones causales.

## Experimentos principales

- Descubrimiento de reglas para hemorragia, neumotórax y el estado derivado sin complicaciones a partir de variables clínicas y geométricas.
- Análisis de robustez frente a la configuración de búsqueda.
- Obtención de reglas de consenso y evaluación de su estabilidad.
- Estudio de la aportación de las variables geométricas mediante comparaciones sobre una cohorte común.
- Comparación exploratoria entre modelos radiómicos generales y modelos entrenados específicamente dentro de los subgrupos descubiertos.

Los identificadores de paciente se utilizan únicamente para alinear las fuentes de información y no forman parte de las reglas.

## Organización

- `run_subgroup_discovery.py`: descubrimiento, validación y análisis de estabilidad de las reglas.
- `launch_subgroup_discovery.sbatch`: ejecución principal mediante SLURM.
- `subgroup_radiomics_ml_20260831/`: modelos radiómicos generales y específicos de subgrupo.
- `tests/`: pruebas del flujo de descubrimiento de subgrupos.

Los resultados completos, las tablas por paciente y los artefactos regenerables se mantienen fuera del control de versiones.