import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt, center_of_mass
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
CLINICAL_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data_limpios_SD.csv"  # tu clinical actual (210)
OUT_GEOM_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/nodule_geometry_210.csv"

ROOT = "/mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/resize_medium/nifti"
LUNG_DIR = os.path.join(ROOT, "masks_lung")
NODULE_DIR = os.path.join(ROOT, "masks_nodule")

ID_COL = "patient_id"  # asegúrate de que tu clinical actual usa patient_id

# -------------------------
# Helpers
# -------------------------
def find_mask_path(dir_, pid):
    cand_gz = os.path.join(dir_, f"{pid}.nii.gz")
    if os.path.exists(cand_gz):
        return cand_gz
    cand_nii = os.path.join(dir_, f"{pid}.nii")
    if os.path.exists(cand_nii):
        return cand_nii
    return None

def load_mask(path):
    img = nib.load(path)
    data = img.get_fdata()
    mask = (data > 0.5).astype(np.uint8)
    spacing = img.header.get_zooms()[:3]  # en el orden de ejes del array
    return mask, spacing

def equivalent_diameter_mm(nodule_mask, spacing):
    # Volumen en mm^3
    voxel_vol = float(spacing[0] * spacing[1] * spacing[2])
    V = float(nodule_mask.sum()) * voxel_vol
    if V <= 0:
        return np.nan
    # diámetro equivalente (esfera) en mm
    return (6.0 * V / np.pi) ** (1.0 / 3.0)

def compute_depths_to_pleura(lung_mask, nodule_mask, spacing):
    """
    Distancia a pleura aproximada como borde del pulmón.
    Para EDT, necesitamos una máscara "inside_lung" = 1 dentro pulmón, 0 fuera.
    La EDT sobre inside_lung devuelve, en voxels dentro del pulmón, distancia al 0 más cercano (fuera),
    es decir distancia a pleura. Con sampling=spacing, queda en mm.
    """
    nod_idx = np.where(nodule_mask > 0)
    if nod_idx[0].size == 0:
        return np.nan, np.nan

    # Detecta si lung_mask está invertida usando el solapamiento con el nódulo
    overlap = lung_mask[nod_idx].mean()  # si es ~1, lung=1 dentro; si ~0, está invertida
    inside_lung = lung_mask if overlap >= 0.5 else (1 - lung_mask)

    edt = distance_transform_edt(inside_lung, sampling=spacing)

    prof_min = float(np.min(edt[nod_idx]))

    c0, c1, c2 = center_of_mass(nodule_mask)  # coordenadas en orden de ejes del array
    c0, c1, c2 = int(round(c0)), int(round(c1)), int(round(c2))
    c0 = np.clip(c0, 0, edt.shape[0] - 1)
    c1 = np.clip(c1, 0, edt.shape[1] - 1)
    c2 = np.clip(c2, 0, edt.shape[2] - 1)

    prof_cent = float(edt[c0, c1, c2])
    return prof_min, prof_cent

# -------------------------
# Main
# -------------------------
df_cli = pd.read_csv(CLINICAL_CSV)

if ID_COL not in df_cli.columns:
    raise ValueError(f"No encuentro la columna {ID_COL} en {CLINICAL_CSV}. Columnas: {list(df_cli.columns)[:30]}")

df_cli[ID_COL] = df_cli[ID_COL].astype(str).str.strip()
patients = df_cli[ID_COL].tolist()

rows = []
missing_lung = 0
missing_nod = 0
shape_mismatch = 0

for pid in tqdm(patients, desc="Calculando tamaño y profundidades"):
    lung_path = find_mask_path(LUNG_DIR, pid)
    nod_path  = find_mask_path(NODULE_DIR, pid)

    if not lung_path:
        missing_lung += 1
    if not nod_path:
        missing_nod += 1

    if (not lung_path) or (not nod_path):
        rows.append({
            ID_COL: pid,
            "tamano_nodulo_mm": np.nan,
            "profundidad_min_pleura_mm": np.nan,
            "profundidad_centroidal_pleura_mm": np.nan,
        })
        continue

    try:
        lung_mask, sp_lung = load_mask(lung_path)
        nodule_mask, sp_nod = load_mask(nod_path)

        if lung_mask.shape != nodule_mask.shape or (not np.allclose(sp_lung, sp_nod, atol=1e-6)):
            shape_mismatch += 1
            rows.append({
                ID_COL: pid,
                "tamano_nodulo_mm": np.nan,
                "profundidad_min_pleura_mm": np.nan,
                "profundidad_centroidal_pleura_mm": np.nan,
            })
            continue

        tam = equivalent_diameter_mm(nodule_mask, sp_nod)
        prof_min, prof_cent = compute_depths_to_pleura(lung_mask, nodule_mask, sp_lung)

        rows.append({
            ID_COL: pid,
            "tamano_nodulo_mm": tam,
            "profundidad_min_pleura_mm": prof_min,
            "profundidad_centroidal_pleura_mm": prof_cent,
        })

    except Exception:
        rows.append({
            ID_COL: pid,
            "tamano_nodulo_mm": np.nan,
            "profundidad_min_pleura_mm": np.nan,
            "profundidad_centroidal_pleura_mm": np.nan,
        })

df_geom = pd.DataFrame(rows)

print("Resumen:")
print("Pacientes:", len(df_geom))
print("Faltan lung masks:", missing_lung)
print("Faltan nodule masks:", missing_nod)
print("Incompatibles (shape/spacing):", shape_mismatch)

print("\nNaNs (%):")
print(df_geom[["tamano_nodulo_mm", "profundidad_min_pleura_mm", "profundidad_centroidal_pleura_mm"]].isna().mean())

df_geom.to_csv(OUT_GEOM_CSV, index=False)
print(f"\nOK -> {OUT_GEOM_CSV}")