import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt, center_of_mass
from tqdm import tqdm


# RUTAS 
ROOT = "/mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/resize_medium/nifti"
LUNG_DIR = os.path.join(ROOT, "masks_lung")
NODULE_DIR = os.path.join(ROOT, "masks_nodule")

CLINICAL_CSV = "../clinical_data/datos_clinicos_limpio_j.csv"      # datos tabulares clinicos
RADIOMICS_CSV = "./data/radiomic_data/radiomics_lung_nodules_more_preprocs.csv"   # radiomica nodulos y lung

OUT_CSV = "clinico_radiomica_con_tamano_y_profundidades.csv"


# utilidades
def find_mask_path(dir_, pid):
    """Devuelve la ruta a la máscara del paciente pid en dir_ aceptando .nii.gz o .nii."""
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
    # binariza por si viene suave/probabilístico
    mask = (data > 0.5).astype(np.uint8)
    spacing = img.header.get_zooms()[:3]  # (sx, sy, sz) en mm
    return mask, spacing, img.shape

def compute_depths_to_pleura(lung_mask, nodule_mask, spacing):
    """
    Profundidad respecto a pleura (tomada como frontera del pulmón):
      - mínima desde cualquier voxel del nódulo
      - desde el centroide del nódulo
    Devuelve (prof_min_mm, prof_centroid_mm)
    """
    # "Exterior" del pulmón: voxeles donde lung==0
    exterior = (lung_mask == 0).astype(np.uint8)
    # EDT con muestreo anisótropo: devuelve mm directamente
    edt = distance_transform_edt(exterior, sampling=spacing)

    nod_idx = np.where(nodule_mask > 0)
    if len(nod_idx[0]) == 0:
        return np.nan, np.nan

    prof_min = float(np.min(edt[nod_idx]))

    cz, cy, cx = center_of_mass(nodule_mask)
    cz, cy, cx = int(round(cz)), int(round(cy)), int(round(cx))
    cz = np.clip(cz, 0, edt.shape[0]-1)
    cy = np.clip(cy, 0, edt.shape[1]-1)
    cx = np.clip(cx, 0, edt.shape[2]-1)
    prof_centroid = float(edt[cz, cy, cx])

    return prof_min, prof_centroid

def compute_size_mm_from_row(row):
    col_max3d = "nodule_original_shape_Maximum3DDiameter"
    col_meshV = "nodule_original_shape_MeshVolume"
    if col_max3d in row and pd.notna(row[col_max3d]):
        return float(row[col_max3d])
    if col_meshV in row and pd.notna(row[col_meshV]) and float(row[col_meshV]) > 0:
        V = float(row[col_meshV])  # mm^3
        return (6.0 * V / np.pi) ** (1.0 / 3.0)  # diámetro equivalente
    return np.nan


# 1) cargar y fusionar CSVs
df_cli = pd.read_csv(CLINICAL_CSV)

df_rad = pd.read_csv(RADIOMICS_CSV)
# si tienes múltiples preprocesados por paciente y quieres uno concreto, filtra aquí por preproc_name.
# Ejemplo:
# df_rad = df_rad[df_rad["preproc_name"] == "resize_cube64_hu_m300_1400"].copy()

# nos quedamos con una fila por paciente si hay duplicados:
if "patient_id" in df_rad.columns:
    df_rad = df_rad.drop_duplicates(subset=["patient_id"], keep="first").copy()
    df_rad["Id_paciente"] = df_rad["patient_id"]
else:
    raise ValueError("El CSV radiómico debe tener la columna 'patient_id'.")

# tamaño del nódulo
df_rad["tamano_nodulo_mm"] = df_rad.apply(compute_size_mm_from_row, axis=1)
df_size = df_rad[["Id_paciente", "tamano_nodulo_mm"]].copy()

df = df_cli.merge(df_size, on="Id_paciente", how="left")

# 2) calcular profundidades para quienes tengan máscaras
prof_min_list = []
prof_cent_list = []

# sólo pacientes presentes en el CSV clínico (acelera)
patients = df["Id_paciente"].astype(str).tolist()

for pid in tqdm(patients, desc="Calculando profundidades pleurales"):
    lung_path = find_mask_path(LUNG_DIR, pid)
    nod_path  = find_mask_path(NODULE_DIR, pid)

    if not lung_path or not nod_path:
        prof_min_list.append(np.nan)
        prof_cent_list.append(np.nan)
        continue

    try:
        lung_mask, sp_lung, shp_lung = load_mask(lung_path)
        nodule_mask, sp_nod, shp_nod = load_mask(nod_path)

        # control de coherencia básica
        same_shape = lung_mask.shape == nodule_mask.shape
        same_spacing = np.allclose(sp_lung, sp_nod, rtol=0, atol=1e-6)
        if not (same_shape and same_spacing):
            # Si tus máscaras vienen del mismo pipeline de resize, normalmente coinciden.
            # Si no, aquí tocaría reamostrar; por simplicidad: NaN.
            prof_min, prof_cent = np.nan, np.nan
        else:
            prof_min, prof_cent = compute_depths_to_pleura(lung_mask, nodule_mask, sp_lung)

    except Exception as e:
        # ante cualquier problema, deja NaN y sigue
        prof_min, prof_cent = np.nan, np.nan

    prof_min_list.append(prof_min)
    prof_cent_list.append(prof_cent)

df["profundidad_min_pleura_mm"] = prof_min_list
df["profundidad_centroidal_pleura_mm"] = prof_cent_list


# 3) guardar
df.to_csv(OUT_CSV, index=False)
print(f"OK -> {OUT_CSV}")
