#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
import pandas as pd
from tqdm import tqdm
import argparse

# Etiquetas esperadas
RUL, RML, RLL, LUL, LLL = 1, 2, 3, 4, 5

PAIRS = {
    "right_horizontal": (RUL, RML),
    "right_oblique":    (RML, RLL),
    "left_oblique":     (LUL, LLL),
}

def load_mask(path):
    nii = nib.load(path)
    mask = nii.get_fdata().astype(np.int16)
    spacing = nii.header.get_zooms()[:3]
    return mask, spacing, nii.affine, nii.header

def mm_to_voxels(mm, spacing):
    return tuple(max(1, int(round(mm/s))) for s in spacing)

def make_barrera(interface, spacing, dilate_mm=3.0):
    rad = mm_to_voxels(dilate_mm, spacing)
    st = generate_binary_structure(3,1)
    barrera = interface.copy()
    for _ in range(max(rad)):
        barrera = binary_dilation(barrera, structure=st)
    return barrera

def find_interface(A, B):
    st = generate_binary_structure(3,1)
    A_edge = np.logical_and(A, binary_dilation(B, structure=st))
    B_edge = np.logical_and(B, binary_dilation(A, structure=st))
    return np.logical_or(A_edge, B_edge)

def flood_reachable(unionAB, seeds, forbidden):
    st = generate_binary_structure(3,1)
    visited = np.zeros_like(unionAB, dtype=bool)
    frontier = np.logical_and(seeds, np.logical_and(unionAB, ~forbidden))
    visited[frontier] = True
    changed = True
    while changed:
        grown = binary_dilation(frontier, structure=st)
        grown = np.logical_and(grown, unionAB)
        grown = np.logical_and(grown, ~forbidden)
        grown = np.logical_and(grown, ~visited)
        changed = grown.any()
        frontier = grown
        visited[grown] = True
    return visited

def has_significant_bridge(reachable, target_mask, spacing, min_bridge_mm3=500):
    bridge_vox = np.logical_and(reachable, target_mask)
    if not bridge_vox.any():
        return False
    voxel_vol = np.prod(spacing)
    vol_mm3 = bridge_vox.sum() * voxel_vol
    return vol_mm3 >= min_bridge_mm3

def test_pair(mask, spacing, labA, labB, erode_mm=0.5, dilate_mm=1.0, min_bridge_mm3=500):
    A = (mask == labA)
    B = (mask == labB)

    er_iters = max(1, int(round(erode_mm/np.mean(spacing))))
    st = generate_binary_structure(3,1)
    if er_iters > 0:
        A = binary_erosion(A, structure=st, iterations=er_iters)
        B = binary_erosion(B, structure=st, iterations=er_iters)

    unionAB = np.logical_or(A, B)
    if not unionAB.any():
        return False, 0.0

    interface = find_interface(A, B)
    barrera = make_barrera(interface, spacing, dilate_mm=dilate_mm)
    seeds = np.logical_and(A, ~barrera)
    if not seeds.any():
        seeds = A.copy()

    reachable = flood_reachable(unionAB, seeds, forbidden=barrera)
    incomplete = has_significant_bridge(reachable, B, spacing, min_bridge_mm3=50)

    voxel_vol = np.prod(spacing)
    bridge_vox = np.logical_and(reachable, B)
    score_mm3 = bridge_vox.sum() * voxel_vol
    return incomplete, score_mm3

def analyze_patient(lobes_nii_path):
    mask, spacing, _, _ = load_mask(lobes_nii_path)
    results = {}
    for name, (la, lb) in PAIRS.items():
        inc, score = test_pair(mask, spacing, la, lb)
        results[f"{name}_incomplete"] = int(inc)
        results[f"{name}_bridge_mm3"] = float(score)
    results["any_incomplete"] = int(any(results[k] for k in results if k.endswith("_incomplete")))
    return results

# ==== batch con escritura online ====
def run_batch(lobes_dir, out_csv):
    nii_files = [fn for fn in os.listdir(lobes_dir)
                 if fn.endswith(".nii") or fn.endswith(".nii.gz")]
    nii_files.sort()
    if not nii_files:
        raise FileNotFoundError(f"No hay NIfTI en {lobes_dir}")

    # Crear CSV con encabezado
    first_file = os.path.join(lobes_dir, nii_files[0])
    first_res = analyze_patient(first_file)
    columns = ["patient_id"] + list(first_res.keys())
    with open(out_csv, "w") as f:
        f.write(",".join(columns) + "\n")

    for fn in tqdm(nii_files, desc="Analizando pacientes", unit="pac", dynamic_ncols=True):
        pid = os.path.splitext(os.path.splitext(fn)[0])[0]
        path = os.path.join(lobes_dir, fn)
        try:
            res = analyze_patient(path)
            res["patient_id"] = pid
            # Escribir cada paciente inmediatamente
            row = [str(res[c]) for c in columns]
            with open(out_csv, "a") as f:
                f.write(",".join(row) + "\n")
            print(f"[INFO] {pid} | any_incomplete={res['any_incomplete']}", flush=True)
        except Exception as e:
            print(f"[ERROR] {pid}: {e}", flush=True)
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lobes_dir", required=True,
                        help="Carpeta con NIfTI de lóbulos (labelmap 1..5 por paciente).")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    # Crear carpeta de salida si no existe
    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)

    # Ejecuta batch
    run_batch(args.lobes_dir, args.out_csv)
    print(f"[DONE] CSV generado: {os.path.abspath(args.out_csv)}")
