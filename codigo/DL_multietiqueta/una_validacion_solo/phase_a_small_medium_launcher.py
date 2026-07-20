import argparse
from pathlib import Path

import phase_a_serial_common as base

PREFIXES_BY_GROUP = {
    'small': ('resize_small',),
    'medium': ('resize_medium',),
    'small_medium': ('resize_small', 'resize_medium'),
}

BATCH_SIZES_BY_SPATIAL_SHAPE = {
    (128, 256, 256): [8, 4],
    (256, 512, 512): [1],
}

VIRTUAL_BATCH_SIZE_BY_SPATIAL_SHAPE = {
    (256, 512, 512): 4,
}

ORIGINAL_DISCOVER_PREPROCESSINGS = base.discover_preprocessings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['resnet18', 'densenet121'], required=True)
    parser.add_argument('--size-group', choices=['small', 'medium', 'small_medium'], required=True)
    return parser.parse_args()


def build_filtered_discover(prefixes):
    def _discover_selected():
        rows = ORIGINAL_DISCOVER_PREPROCESSINGS()
        selected = [row for row in rows if any(row['name'].startswith(prefix) for prefix in prefixes)]
        if not selected:
            raise RuntimeError(f'No preprocessings found for prefixes: {prefixes}')
        return selected

    return _discover_selected


def candidate_batch_sizes_from_shape(shape):
    spatial = base.spatial_shape(shape)
    if spatial not in BATCH_SIZES_BY_SPATIAL_SHAPE:
        return []
    return list(BATCH_SIZES_BY_SPATIAL_SHAPE[spatial])


def grad_accum_steps_from_shape(shape, batch_size):
    spatial = base.spatial_shape(shape)
    virtual_batch_size = VIRTUAL_BATCH_SIZE_BY_SPATIAL_SHAPE.get(spatial)
    if virtual_batch_size is None:
        return 1
    return max(1, int(virtual_batch_size) // int(batch_size))


def main():
    args = parse_args()
    prefixes = PREFIXES_BY_GROUP[args.size_group]
    base.discover_preprocessings = build_filtered_discover(prefixes)
    base.candidate_batch_sizes_from_shape = candidate_batch_sizes_from_shape
    base.grad_accum_steps_from_shape = grad_accum_steps_from_shape
    base.OUTPUT_ROOT = base.PROJECT_ROOT / 'codigo/DL_multietiqueta/phase_a_small_medium_runs' / args.size_group
    Path(base.OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    base.run_phase_a(args.model)


if __name__ == '__main__':
    main()
