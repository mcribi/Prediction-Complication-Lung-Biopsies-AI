import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import posthoc_gradcam_multilabel as mod

model_name = 'resnet18'
run_root = mod.RUN_ROOTS[model_name]
labels_df = mod.load_labels_df_for_run(run_root)
summary = mod.summarize_run(model_name, run_root)
row = summary['top_df'].iloc[0]
cfg_name = f"{row['preprocessing']}__{row['input_name']}__bs{int(row['batch_size'])}"
fold_df = pd.read_csv(run_root / cfg_name / 'fold_summary.csv')
best_fold_row = fold_df.sort_values(['best_val_f1_micro', 'best_epoch'], ascending=[False, False]).iloc[0]
best_fold = int(best_fold_row['fold'])
val_df = pd.read_csv(run_root / cfg_name / f'fold_{best_fold}' / 'val_predictions_best_epoch.csv')
case = mod.select_cases_for_label(val_df, mod.TARGET_LABELS[0])[0]
pid = case['patient_id']
label = case['label']
case_kind = case['case_kind']
x = mod.load_case_tensor(labels_df, pid, str(row['preprocessing']), str(row['input_name']))
volume = mod.load_display_volume(str(row['preprocessing']), pid)
ref_mask = mod.load_reference_mask(str(row['preprocessing']), str(row['input_name']), pid)
model = mod.build_model_for_config(model_name, x)
state = torch.load(run_root / cfg_name / f'fold_{best_fold}' / 'best_model.pt', map_location='cpu')
model.load_state_dict(state)
_, layer = mod.get_last_conv3d(model)
cam = mod.compute_gradcam(model, layer, x, mod.TARGET_LABELS.index(label), torch.device('cpu'))
mod.validate_aligned_shapes(volume, cam, ref_mask)
out_dir = Path('/mnt/homeGPU/mcribilles/tfm/codigo/DL_multietiqueta/posthoc_gradcam_smoke')
out_dir.mkdir(parents=True, exist_ok=True)
stem = f"{pid}__{mod.LABEL_SAFE[label]}__{case_kind}"
img = mod.build_contact_sheet_array(
    volume,
    cam,
    [
        f'patient={pid}',
        f'label={label}',
        f'display=aligned_preprocessed_ct threshold={mod.CAM_DISPLAY_THRESHOLD:.2f}',
    ],
    ref_mask=ref_mask,
)
img_path = mod.save_rgb_image(img, out_dir / f'{stem}.png')
cut_paths = mod.save_plane_cut_series(volume, cam, out_dir, stem, ref_mask=ref_mask)
arr_np = np.array(Image.open(img_path))
print(json.dumps({
  'img_path': str(img_path),
  'cut_paths': cut_paths[:3],
  'shape': list(arr_np.shape),
  'pixel_mean': float(arr_np.mean()),
  'pixel_max': int(arr_np.max()),
  'cam_max': float(cam.max()),
  'cam_sum': float(cam.sum()),
  'volume_nonzero': int((volume > 0).sum()),
  **mod.cam_overlap_metrics(cam, ref_mask),
}, ensure_ascii=False))
