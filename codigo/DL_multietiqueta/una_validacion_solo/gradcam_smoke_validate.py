import json
from pathlib import Path

import pandas as pd
import torch

import posthoc_gradcam_multilabel as mod

MODEL_NAME = 'resnet18'
TARGET_CFG = 'resize_cube128__nodule_only_masked_ct__bs8'
TARGET_LABEL = 'Hemorragia'
TARGET_PATIENT = '140HASNh'

run_root = mod.RUN_ROOTS[MODEL_NAME]
labels_df = mod.load_labels_df_for_run(run_root)
cfg_dir = run_root / TARGET_CFG
fold_df = pd.read_csv(cfg_dir / 'fold_summary.csv')
best_fold_row = fold_df.sort_values(['best_val_f1_micro', 'best_epoch'], ascending=[False, False]).iloc[0]
best_fold = int(best_fold_row['fold'])
row = pd.Series({
    'preprocessing': TARGET_CFG.split('__')[0],
    'input_name': TARGET_CFG.split('__')[1],
    'batch_size': int(TARGET_CFG.split('__bs')[1]),
})

x = mod.load_case_tensor(labels_df, TARGET_PATIENT, str(row['preprocessing']), str(row['input_name']))
volume = mod.load_display_volume(str(row['preprocessing']), TARGET_PATIENT)
ref_mask = mod.load_reference_mask(str(row['preprocessing']), str(row['input_name']), TARGET_PATIENT)
model = mod.build_model_for_config(MODEL_NAME, x)
state = torch.load(cfg_dir / f'fold_{best_fold}' / 'best_model.pt', map_location='cpu')
model.load_state_dict(state)
_, layer = mod.get_last_conv3d(model)
cam = mod.compute_gradcam(model, layer, x, mod.TARGET_LABELS.index(TARGET_LABEL), torch.device('cpu'))

out_dir = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/posthoc_gradcam_smoke_validate')
out_dir.mkdir(parents=True, exist_ok=True)
stem = f'{TARGET_PATIENT}__{mod.LABEL_SAFE[TARGET_LABEL]}__smoke'
header = [
    f'model={MODEL_NAME} cfg={TARGET_CFG}',
    f'patient={TARGET_PATIENT} label={TARGET_LABEL} fold={best_fold}',
    f'cam_sum={float(cam.sum()):.4f}',
    f'display=aligned_preprocessed_ct threshold={mod.CAM_DISPLAY_THRESHOLD:.2f} mask_outline=green',
]
img = mod.build_contact_sheet_array(volume, cam, header, ref_mask=ref_mask)
img_path = mod.save_rgb_image(img, out_dir / f'{stem}.png')
cut_paths = mod.save_plane_cut_series(volume, cam, out_dir, stem, ref_mask=ref_mask)

result = {
    'img_path': str(img_path),
    'cut_paths': cut_paths,
    'cam_sum': float(cam.sum()),
    'volume_nonzero_frac': float((volume > 1e-6).mean()),
    'refmask_nonzero_frac': float((ref_mask > 0).mean()) if ref_mask is not None else None,
    **mod.cam_overlap_metrics(cam, ref_mask),
}
print(json.dumps(result, ensure_ascii=False))
