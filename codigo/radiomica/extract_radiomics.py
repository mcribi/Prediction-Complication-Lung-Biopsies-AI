import os
import argparse
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ---------- Helpers ----------
def list_dirs(path):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def list_files_by_extension(folder, exts):
    if not os.path.isdir(folder):
        return []
    exts = tuple(e.lower() for e in exts)
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])

def strip_nii_extensions(filename):
    for ext in (".nii.gz", ".nii"):
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename

def ensure_scalar_images(img):
    """
    Devuelve lista de imágenes escalares 3D:
      - Si la imagen es multicanal (vector), separa canales con VectorIndexSelectionCast.
      - Si es 4D (por ejemplo temporal), extrae cada índice temporal con Extract.
      - Si ya es 3D escalar, la devuelve en una lista.
    """
    # Multicanal (vector image)
    if img.GetNumberOfComponentsPerPixel() > 1:
        channels = []
        for c in range(img.GetNumberOfComponentsPerPixel()):
            channels.append(sitk.VectorIndexSelectionCast(img, c))
        return channels

    # 4D (e.g., temporal)
    if img.GetDimension() == 4:
        size = list(img.GetSize())      # [x, y, z, t]
        idx  = [0, 0, 0, 0]
        size_3d = size[:3] + [0]        # extraer 3D en cada t
        out = []
        for t in range(size[3]):
            idx[3] = t
            out.append(sitk.Extract(img, size_3d, idx))
        return out

    # 3D escalar
    return [img]

def filter_feature_dict(d, prefix=None):
    """
    De PyRadiomics nos quedamos sólo con claves que empiezan por 'original_' para evitar
    parámetros/diagnósticos. Opcionalmente añade prefijo (p.ej. 'lung_' o 'nodule_').
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.startswith("original_"):
            out[(prefix + "_" + k) if prefix else k] = v
    return out

def pair_by_basename(images_dir, masks_dir):
    """
    Empareja archivos por nombre base (sin extensión). Devuelve lista de tuplas (patient_id, image_path, mask_path).
    """
    img_files = list_files_by_extension(images_dir, [".nii.gz", ".nii"])
    msk_files = list_files_by_extension(masks_dir,  [".nii.gz", ".nii"])
    img_map = {strip_nii_extensions(f): f for f in img_files}
    msk_map = {strip_nii_extensions(f): f for f in msk_files}
    keys = sorted(set(img_map.keys()) & set(msk_map.keys()))
    pairs = []
    for k in keys:
        pairs.append( (k, os.path.join(images_dir, img_map[k]), os.path.join(masks_dir, msk_map[k])) )
    return pairs

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Extraer radiómicas (PyRadiomics) de varios preprocesados.")
    ap.add_argument("--base_preproc_dir", type=str, required=True,
                    help="Carpeta base con preprocesados (ej: ./volumenes_preprocesados)")
    ap.add_argument("--preproc_names", type=str, nargs="*", default=None,
                    help="Nombres concretos de preprocesado a procesar (si no se da, procesa todos).")
    ap.add_argument("--which_masks", type=str, choices=["lung", "nodule", "both"], default="both",
                    help="Qué máscaras usar.")
    ap.add_argument("--label", type=int, default=1, help="Etiqueta (valor) de la ROI en la máscara.")
    ap.add_argument("--no_resample", action="store_true",
                    help="No re-muestrear (usa el spacing nativo).")
    ap.add_argument("--out_csv", type=str, default="./data/radiomic_data/radiomics_lung_nodules_more_preprocs.csv",
                    help="Ruta del CSV de salida combinado.")
    args = ap.parse_args()

    base_dir = args.base_preproc_dir
    preprocs = args.preproc_names if args.preproc_names else list_dirs(base_dir)
    if not preprocs:
        print(f"[AVISO] No hay preprocesados en {base_dir}")
        return

    # Parámetros PyRadiomics
    params = {
        "binWidth": 25,
        "resampledPixelSpacing": None if args.no_resample else [1, 1, 1],
        "interpolator": "sitkBSpline",
        "enableCExtensions": True,
        "normalize": True,
        "removeOutliers": 3,
        "verbose": True,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    rows = []
    for preproc_name in preprocs:
        preproc_path = os.path.join(base_dir, preproc_name)
        # Esperada estructura: <preproc>/nifti/{images, masks_lung, masks_nodule}
        nifti_dir = os.path.join(preproc_path, "nifti")
        images_dir = os.path.join(nifti_dir, "images")
        masks_lung_dir   = os.path.join(nifti_dir, "masks_lung")
        masks_nodule_dir = os.path.join(nifti_dir, "masks_nodule")

        if not os.path.isdir(images_dir):
            print(f"[AVISO] {preproc_name}: no existe {images_dir}. Saltando.")
            continue

        kinds = []
        if args.which_masks in ("lung", "both") and os.path.isdir(masks_lung_dir):
            kinds.append(("lung", masks_lung_dir))
        if args.which_masks in ("nodule", "both") and os.path.isdir(masks_nodule_dir):
            kinds.append(("nodule", masks_nodule_dir))

        if not kinds:
            print(f"[AVISO] {preproc_name}: no hay máscaras solicitadas. Saltando.")
            continue

        print(f"\n== Procesando preproc: {preproc_name} ==")
        for kind, mdir in kinds:
            pairs = pair_by_basename(images_dir, mdir)
            print(f"  {kind}: {len(pairs)} parejas imagen-máscara encontradas.")
            for pid, img_path, msk_path in tqdm(pairs, desc=f"{preproc_name} [{kind}]"):
                try:
                    img = sitk.ReadImage(img_path)
                    msk = sitk.ReadImage(msk_path)

                    # Asegurar lista de 3D escalares
                    scalars = ensure_scalar_images(img)
                    if len(scalars) == 1:
                        feats = extractor.execute(scalars[0], msk, label=args.label)
                        row = {"preproc_name": preproc_name, "patient_id": pid, "mask_kind": kind}
                        row.update(filter_feature_dict(feats, prefix=kind))
                        rows.append(row)
                    else:
                        combined = {"preproc_name": preproc_name, "patient_id": pid, "mask_kind": kind}
                        for ci, simg in enumerate(scalars):
                            feats = extractor.execute(simg, msk, label=args.label)
                            combined.update(filter_feature_dict(feats, prefix=f"{kind}_ch{ci}"))
                        rows.append(combined)

                except Exception as e:
                    print(f"  [ERROR] {preproc_name}::{pid} [{kind}]: {e}")

    # Guardar
    if rows:
        df = pd.DataFrame(rows)
        base_cols = ["preproc_name", "patient_id", "mask_kind"]
        feat_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + sorted(feat_cols)]
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"\nGuardado: {args.out_csv} | filas={len(df)} | n_feats={df.shape[1]-len(base_cols)}")
    else:
        print("\nNo se generaron features.")

if __name__ == "__main__":
    main()
