#!/bin/bash
set -euo pipefail

ROOT=/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/two_outputs_derived
LAUNCH_EPOCH=${LAUNCH_EPOCH:-$(date +%s)}
BATCH_ID=${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}
REGISTRY_DIR="$ROOT/submitted_batches"
REGISTRY="$REGISTRY_DIR/${BATCH_ID}.env"
mkdir -p "$REGISTRY_DIR"
printf 'BATCH_ID=%q\nLAUNCH_EPOCH=%q\n' "$BATCH_ID" "$LAUNCH_EPOCH" > "$REGISTRY"
record_job() {
  local key="$1" value="$2"
  printf '%s=%q\n' "$key" "$value" >> "$REGISTRY"
  echo "$key=$value"
}
trap 'printf "SUBMISSION_STATUS=failed_at_line_%q\n" "$LINENO" >> "$REGISTRY"' ERR
if [[ -n "${SMOKE_JOB_ID:-}" ]]; then
  SMOKE_JOB="$SMOKE_JOB_ID"
else
  SMOKE_JOB=$(sbatch --parsable "$ROOT/code/slurm/submit_two_outputs_smoke.sbatch")
fi
record_job SMOKE_JOB "$SMOKE_JOB"

IMAGE_JOB=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" "$ROOT/code/slurm/submit_image_two_outputs_array.sbatch")
record_job IMAGE_JOB "$IMAGE_JOB"
MULTIMODAL_JOB=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" "$ROOT/code/slurm/submit_multimodal_two_outputs_array.sbatch")
record_job MULTIMODAL_JOB "$MULTIMODAL_JOB"
PRETRAINED_DIRECT_JOB=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" "$ROOT/code/slurm/submit_pretrained_two_outputs_array.sbatch")
record_job PRETRAINED_DIRECT_JOB "$PRETRAINED_DIRECT_JOB"
PRETRAINED_FREEZE_JOB=$(sbatch --parsable --dependency="afterok:${SMOKE_JOB}" "$ROOT/code/slurm/submit_pretrained_freeze_two_outputs_array.sbatch")
record_job PRETRAINED_FREEZE_JOB "$PRETRAINED_FREEZE_JOB"
COLLECTOR_JOB=$(sbatch --parsable \
  --export="ALL,MIN_MTIME_EPOCH=${LAUNCH_EPOCH}" \
  --dependency="afterany:${IMAGE_JOB}:${MULTIMODAL_JOB}:${PRETRAINED_DIRECT_JOB}:${PRETRAINED_FREEZE_JOB}" \
  "$ROOT/code/slurm/submit_collect_two_output_results.sbatch")
record_job COLLECTOR_JOB "$COLLECTOR_JOB"
printf 'SUBMISSION_STATUS=complete\n' >> "$REGISTRY"

printf '%s\n' "$SMOKE_JOB" "$IMAGE_JOB" "$MULTIMODAL_JOB" "$PRETRAINED_DIRECT_JOB" "$PRETRAINED_FREEZE_JOB" "$COLLECTOR_JOB" > "$ROOT/submitted_job_ids.txt"
squeue -j "$SMOKE_JOB,$IMAGE_JOB,$MULTIMODAL_JOB,$PRETRAINED_DIRECT_JOB,$PRETRAINED_FREEZE_JOB,$COLLECTOR_JOB" -o '%.20i %.12P %.25j %.8T %.12M %.6D %R'
