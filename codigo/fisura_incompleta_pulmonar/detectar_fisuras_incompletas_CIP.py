#!/usr/bin/env python3
import os
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm

# CIP
import sys
sys.path.insert(0, "/mnt/homeGPU/mcribilles/tfm/codigo/CIP/cip_python")
from fissure_analysis import compute_fissure_completeness

#config
CT_ROOT = "/mnt/homeGPU/mcribilles/TFG/volumenes/nifti_convertidos_anonimizados"
LOBES_ROOT = "/mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/lobes_masks_from_dicom"
OUT_CSV = "/mnt/homeGPU/mcribilles/tfm/codigo/fisura_incompleta_pulmonar/fisuras_flags_CIP.csv"

# Crear lista de pacientes
ct_files = sorted([p for p in Path(CT_ROOT).iterdir() if p.suffix in [".nii", ".nii.gz"]])

results = []

for ct_path in tqdm(ct_files, desc="Analizando pacientes", unit="pac"):
    pid = ct_path.stem
    lobes_path = Path(LOBES_ROOT) / f"{pid}_lobes.nii.gz"

    if not lobes_path.exists():
        print(f"[WARN] {pid}: falta segmentación de lóbulos")
        continue

    try:
        print(f"[INFO] Procesando paciente {pid}...")

        ct_img = sitk.ReadImage(str(ct_path))
        lobes_img = sitk.ReadImage(str(lobes_path))

        # CIP: calcular completitud de fisuras
        fissure_scores = compute_fissure_completeness(ct_img, lobes_img)

        row = {"patient_id": pid}
        row.update(fissure_scores)
        row["any_incomplete"] = int(any(v < 1.0 for v in fissure_scores.values()))

        results.append(row)
        print(f"[OK] {pid} -> any_incomplete={row['any_incomplete']}")

    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        continue

# Guardar CSV final
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print(f"[DONE] Resultados guardados en {OUT_CSV}")
