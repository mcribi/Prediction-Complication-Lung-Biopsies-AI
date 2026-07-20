from pathlib import Path

import phase_a_serial_common as base

TARGET_PREPROC = "resize_medium"
TARGET_INPUT = "lung_only_masked_ct"
TARGET_MODEL = "resnet18"

ORIGINAL_DISCOVER_PREPROCESSINGS = base.discover_preprocessings


def discover_one_preprocessing():
    rows = ORIGINAL_DISCOVER_PREPROCESSINGS()
    selected = [row for row in rows if row["name"] == TARGET_PREPROC]
    if not selected:
        raise RuntimeError(f"Preprocessing not found: {TARGET_PREPROC}")
    return selected


def candidate_batch_sizes_from_shape(shape):
    spatial = base.spatial_shape(shape)
    if spatial == (256, 512, 512):
        return [1]
    return []


def grad_accum_steps_from_shape(shape, batch_size):
    spatial = base.spatial_shape(shape)
    if spatial == (256, 512, 512):
        return 4
    return 1


def main():
    base.discover_preprocessings = discover_one_preprocessing
    base.candidate_batch_sizes_from_shape = candidate_batch_sizes_from_shape
    base.grad_accum_steps_from_shape = grad_accum_steps_from_shape
    base.PRIMARY_INPUTS = [cfg for cfg in base.PRIMARY_INPUTS if cfg["name"] == TARGET_INPUT]
    if not base.PRIMARY_INPUTS:
        raise RuntimeError(f"Input config not found: {TARGET_INPUT}")
    base.OUTPUT_ROOT = Path("/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/phase_a_smoke_tests/resnet18_medium_dgx2_20260717_111757")
    base.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    base.run_phase_a(TARGET_MODEL)


if __name__ == "__main__":
    main()
