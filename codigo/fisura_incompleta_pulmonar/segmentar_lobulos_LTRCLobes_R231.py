#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from lungmask import mask

# Configuración
NIFTI_ROOT = "/mnt/homeGPU/mcribilles/TFG/volumenes/nifti_convertidos_anonimizados"  # carpeta raíz con subcarpetas de pacientes
OUT_DIR = "/mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/lobes_masks_from_dicom_LTRCLobes_R231"  # salida NIfTI
MODEL_NAME = "R231"


os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Cargar modelo
# ----------------------------
print(f"[INFO] Cargando modelo {MODEL_NAME}...")
model = mask.get_model(MODEL_NAME)
print("[INFO] Modelo cargado.")

# ----------------------------
# Buscar todos los NIfTI recursivamente
# ----------------------------
nii_files = sorted(Path(NIFTI_ROOT).rglob("*.nii*"))
if not nii_files:
    raise FileNotFoundError(f"No se encontraron NIfTI en {NIFTI_ROOT}")

# ----------------------------
# Procesar cada paciente
# ----------------------------
for nii_path in tqdm(nii_files, desc="Segmentando lobos", unit="pac", dynamic_ncols=True):
    pid = nii_path.stem
    out_path = Path(OUT_DIR) / f"{pid}_lobes.nii.gz"

    if out_path.exists():
        print(f"[SKIP] {pid}: ya existe {out_path.name}")
        continue

    try:
        print(f"[INFO] Procesando paciente {pid}...")

        # Cargar imagen con nibabel
        nii_img = nib.load(str(nii_path))

        # Aplicar segmentación (devuelve un Nifti1Image)
        seg_nii = mask.apply(nii_img, model=model)

        # Guardar resultado
        nib.save(seg_nii, str(out_path))
        print(f"[OK] {pid}: guardado {out_path.name}")

    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        continue

print("[DONE] Segmentación de todos los pacientes completada.")