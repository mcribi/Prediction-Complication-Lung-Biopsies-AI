# Experimentos DL con dos salidas

## Objetivos

- Entrenar unicamente `Hemorragia` y `Neumotorax` como salidas independientes.
- Derivar `Sin_complicacion` como `(Hemorragia == 0) and (Neumotorax == 0)`.
- Guardar sus metricas binarias, pero no atribuirle una probabilidad independiente: ROC-AUC y average precision quedan como no definidas (`NaN`).
- Conservar arquitectura, entradas, preprocesado, batch, optimizador, scheduler, semillas, folds, epocas y paciencia de los experimentos originales.
- Comparar el umbral fijo 0.5 con umbrales por etiqueta seleccionados solo en `val_inner`.

## Protocolo de umbrales

1. El checkpoint se selecciona con el criterio historico de F1 micro a 0.5 en `val_inner`.
2. Una vez fijado el checkpoint, se calculan sus probabilidades en `val_inner`.
3. Para cada etiqueta se explora la rejilla 0.10, 0.15, ..., 0.90.
4. Se maximiza F1 de la etiqueta. Los empates se resuelven por cercania a 0.5 y despues por menor umbral.
5. Los dos umbrales se congelan y se aplican una sola vez a `test_outer`.
6. Tambien se guardan las metricas de referencia con 0.5.

`test_outer` no participa en la eleccion del checkpoint ni de los umbrales.

## Artefactos por fold

- `history.csv`
- `best_model.pt`
- `split_assignments.csv`
- `threshold_search_curves.csv`
- `selected_thresholds.csv`
- `val_predictions_best_epoch.csv`
- `test_predictions_best_epoch.csv`
- `test_metrics_fixed_0_5.json`
- `test_metrics_tuned.json`

Las predicciones incluyen las dos salidas, los umbrales y el estado derivado.

## Artefactos agregados

- `fold_summary.csv`
- `oof_predictions.csv`
- `oof_metrics_fixed_0_5.json`
- `oof_metrics_tuned.json`
- tablas con media y desviacion estandar muestral (`ddof=1`) entre los cinco folds.

No se interpretara ninguna configuracion sin los cinco folds externos completos y sin que cada paciente aparezca exactamente una vez en OOF.

## Cohorte

El codigo exige 210 pacientes en la tabla clinica fuente. El helper historico excluye a `27HASD`, que presenta `Derrame pleural` como unica complicacion fuera de las dos dianas, por lo que quedan 209 pacientes elegibles. Tras exigir disponibilidad completa de imagen, la cohorte efectiva esperada es de 204 pacientes.

## Transferencia y ajuste fino

- Pilotos directos: MedicalNet ResNet-18 y Models Genesis.
- `phase2`: MedicalNet ResNet-34 y Models Genesis, dos configuraciones por fuente.
- `ct_simple`: dos entradas por fuente con batch 2, elegido tras los OOM historicos con batch 4.
- `luna16`: ResNet-34 con dos entradas y el ultimo checkpoint supervisado de nodulos disponible.
- El checkpoint LUNA16 se ha fijado por ruta y SHA-256 para que sus dos configuraciones usen exactamente la misma inicializacion.
- En `freeze/unfreeze`, el encoder permanece congelado cinco epocas, incluidas sus estadisticas BatchNorm, y despues se descongela por completo.
- La carga se considera valida solo si incluye la primera convolucion y al menos diez tensores compatibles.

## Lanzamiento

Los trabajos largos usan `dgx2,dgx`, una GPU por tarea y `4-00:00:00`. Cada array tiene un limite propio de concurrencia, pero el limite global se debe ajustar segun la ocupacion real de `dgx2,dgx` inmediatamente antes del lanzamiento. Todos los barridos se deben enviar con dependencia `afterok` respecto al smoke test.

El launcher `submit_verified_core_jobs.sh` acepta `SMOKE_JOB_ID=<jobid>` para reutilizar un smoke ya completado; si no se proporciona, envia uno nuevo.

Cada lanzamiento registra los JobID de forma incremental en `submitted_batches/`. El collector recibe el instante de inicio del lote y rechaza configuraciones que mezclen artefactos anteriores y actuales.
