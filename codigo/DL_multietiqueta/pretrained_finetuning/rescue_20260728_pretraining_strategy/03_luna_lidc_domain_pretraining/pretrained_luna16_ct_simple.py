from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from monai.networks.nets import resnet18, resnet34

SCRIPT_DIR = Path(__file__).resolve().parent
DL_ROOT = SCRIPT_DIR.parents[2]
CT_SIMPLE_DIR = DL_ROOT / 'pretrained_finetuning' / 'rescue_20260728_pretraining_strategy' / '01_ct_simple_pretrained'
for extra_path in [SCRIPT_DIR, DL_ROOT, CT_SIMPLE_DIR]:
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import pretrained_ct_simple as pilot

DEFAULT_LUNA_RUN_ROOT = DL_ROOT / 'pretrained_finetuning' / 'runs' / 'luna16_nodule_pretraining_real'


def find_latest_luna_checkpoint() -> Path:
    candidates = sorted(DEFAULT_LUNA_RUN_ROOT.glob('*/best_encoder_state_dict.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No LUNA16 checkpoint found under {DEFAULT_LUNA_RUN_ROOT}. Set --luna-checkpoint explicitly or run submit_luna16_nodule_pretrain.sbatch first.')
    return candidates[0]


def build_luna_model(pretraining: str, in_channels: int, out_channels: int, report_dir: Path) -> nn.Module:
    if pretraining == 'luna16_resnet18':
        model = resnet18(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels, shortcut_type='A')
    elif pretraining == 'luna16_resnet34':
        model = resnet34(spatial_dims=3, n_input_channels=in_channels, num_classes=out_channels, shortcut_type='A')
    else:
        return ORIGINAL_BUILD_PRETRAINED_MODEL(pretraining, in_channels, out_channels, report_dir)
    model.fc = nn.Sequential(nn.Dropout(pilot.DROPOUT), model.fc)
    report = pilot.load_matching_weights(model, LUNA_CHECKPOINT, ['conv1.weight'], skip_prefixes=('fc',))
    report.update({
        'pretraining': pretraining,
        'source': 'luna16_supervised_nodule_pretraining',
        'luna_checkpoint': str(LUNA_CHECKPOINT),
        'in_channels': int(in_channels),
        'out_channels': int(out_channels),
        'note': 'fc is reinitialized for TFM multilabel output; conv1 is adapted if project input has more than one channel.',
    })
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'pretrained_loading_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return model


def get_head_and_encoder_params(model: nn.Module, pretraining: str):
    if pretraining in {'medicalnet', 'luna16_resnet18', 'luna16_resnet34'}:
        head_params = list(model.fc.parameters())
    else:
        head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    return encoder_params, head_params


def patch_pilot_module() -> None:
    pilot.build_pretrained_model = build_luna_model
    pilot.get_head_and_encoder_params = get_head_and_encoder_params


def main() -> None:
    parser = argparse.ArgumentParser(description='TFM nested-CV fine tuning from a bounded LUNA16 nodule-pretrained ResNet checkpoint.')
    parser.add_argument('--pretraining', choices=['luna16_resnet18', 'luna16_resnet34'], default='luna16_resnet34')
    parser.add_argument('--luna-checkpoint', type=Path, default=None)
    parser.add_argument('--run-tag', default='luna16_domain_pretrain_transfer')
    parser.add_argument('--config-index', type=int, action='append', default=None, help='0-based config index from pretrained_ct_simple.PILOT_CONFIGS; repeat to run a subset')
    parser.add_argument('--batch-size-override', type=int, default=None)
    args = parser.parse_args()

    global LUNA_CHECKPOINT
    LUNA_CHECKPOINT = args.luna_checkpoint if args.luna_checkpoint is not None else find_latest_luna_checkpoint()
    if not LUNA_CHECKPOINT.exists():
        raise FileNotFoundError(f'LUNA16 checkpoint not found: {LUNA_CHECKPOINT}')
    patch_pilot_module()
    pilot.run_pilot(args.pretraining, run_tag=args.run_tag, config_indices=args.config_index, batch_size_override=args.batch_size_override)


ORIGINAL_BUILD_PRETRAINED_MODEL = pilot.build_pretrained_model
LUNA_CHECKPOINT = None

if __name__ == '__main__':
    main()
