#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import nibabel as nib

# ---------------------------
# Etiquetas finales (labelmap)
# ---------------------------
LABELS = {
    "RUL": 1,
    "RML": 2,
    "RLL": 3,
    "LUL": 4,
    "LLL": 5,
}

# Posibles nombres de archivos que distintas versiones de TS pueden generar
LOBAR_FILENAME_ALIASES: Dict[str, List[str]] = {
    "RUL": ["lung_upper_lobe_right", "right_upper_lobe", "RUL", "lobes_right_upper"],
    "RML": ["lung_middle_lobe_right", "right_middle_lobe", "RML", "lobes_right_middle"],
    "RLL": ["lung_lower_lobe_right", "right_lower_lobe", "RLL", "lobes_right_lower"],
    "LUL": ["lung_upper_lobe_left", "left_upper_lobe", "LUL", "lobes_left_upper"],
    "LLL": ["lung_lower_lobe_left", "left_lower_lobe", "LLL", "lobes_left_lower"],
}

def is_dicom_file(path: Path) -> bool:
    """Detección simple de DICOM (sin pydicom): chequea 'DICM' en byte 128."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            tag = f.read(4)
        return tag == b"DICM"
    except Exception:
        return False

def find_best_dicom_series(patient_dir: Path, min_files: int = 30) -> Optional[Path]:
    """
    Busca recursivamente la subcarpeta con más ficheros DICOM (heurística).
    Devuelve la ruta de esa serie o None si no encuentra ninguna razonable.
    """
    best_dir, best_count = None, 0
    for root, dirs, files in os.walk(patient_dir):
        # ignorar carpetas que claramente no son series
        name = os.path.basename(root).lower()
        if name in {"dicomdir", "lockfile", "version"}:
            continue
        # candidatos: muchos ficheros sin extensión / .dcm
        cand_files = []
        for fn in files:
            p = Path(root) / fn
            # rápido: ignora ficheros diminutos
            if p.is_file() and p.stat().st_size > 2048:
                # no te fíes de la extensión; valida DICOM por magic
                if is_dicom_file(p):
                    cand_files.append(p)
        if len(cand_files) > best_count:
            best_count = len(cand_files)
            best_dir = Path(root)

    if best_dir is not None and best_count >= min_files:
        print(f"[INFO] Serie elegida: {best_dir} ({best_count} ficheros DICOM)")
        return best_dir
    return None

def run_cmd(cmd: List[str], fail_msg: str) -> subprocess.CompletedProcess:
    print(f"[CMD] {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] {fail_msg}", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
    return res

def run_totalsegmentator(
    image_path: Path,
    case_out_dir: Path,
    fast: bool = True,
    device: str = "gpu",
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Llama a TotalSegmentator (task=total + lobes via --roi_subset).
    Devuelve (ok, stderr_text).
    """
    cmd = [
        "TotalSegmentator",
        "-i", str(image_path),
        "-o", str(case_out_dir),
        "-ta", "total",
        "--roi_subset",
        "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    ]
    # si quieres salida multi-label directa:
    cmd.append("-ml")

    if fast:
        cmd.append("--fast")
    if device in ("cpu", "gpu"):
        cmd += ["--device", device]
    if extra_args:
        cmd += extra_args
    
    log_file = case_out_dir / "totalsegmentator.log"  # Log de salida
    with open(log_file, "w") as log:
        print(f"[INFO] Ejecutando TotalSegmentator sobre: {image_path}", file=log)
        res = subprocess.run(cmd, stdout=log, stderr=log, text=True, timeout=timeout)
    
    if res.returncode != 0:
        print(f"[ERROR] TotalSegmentator falló para {image_path.name}", file=sys.stderr)
        with open(log_file, "r") as log:
            print(log.read())
        return False, "Error en TotalSegmentator"
    
    print(f"[INFO] TotalSegmentator completado para {image_path.name}", file=sys.stderr)
    return True, "Proceso completado con éxito"


def convert_dicom_to_nifti(series_dir: Path, out_dir: Path) -> Optional[Path]:
    """
    Convierte una serie DICOM a NIfTI usando dcm2niix. Devuelve la ruta del NIfTI generado.
    """
    # comprobación rápida de dcm2niix en PATH
    chk = shutil.which("dcm2niix")
    if chk is None:
        print("[WARN] dcm2niix no encontrado en PATH. No se puede convertir DICOM -> NIfTI.", file=sys.stderr)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["dcm2niix", "-z", "y", "-o", str(out_dir), "-f", "img", str(series_dir)]
    res = run_cmd(cmd, fail_msg=f"dcm2niix falló para {series_dir}")
    if res.returncode != 0:
        return None

    # recoge el primer .nii.gz generado
    cand = list(out_dir.glob("img*.nii.gz"))
    if not cand:
        cand = list(out_dir.glob("*.nii.gz"))
    if cand:
        print(f"[INFO] DICOM -> NIfTI: {cand[0]}")
        return cand[0]
    print("[ERROR] dcm2niix completó pero no se encontró NIfTI de salida.", file=sys.stderr)
    return None

def find_single_lobemap(out_dir: Path) -> Optional[Path]:
    candidates = [
        out_dir / "lung_lobes.nii.gz",
        out_dir / "lung_lobes.nii",
        out_dir / "segmentation_lung_lobes.nii.gz",
        out_dir / "segmentation_lung_lobes.nii",
    ]
    for c in candidates:
        if c.exists():
            return c
    gen = out_dir / "segmentation.nii.gz"
    if gen.exists():
        return gen
    return None

def find_binary_masks(out_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    nifti_paths = list(out_dir.rglob("*.nii")) + list(out_dir.rglob("*.nii.gz"))
    by_stem = {p.stem: p for p in nifti_paths}

    def match_alias(stem: str, alias: str) -> bool:
        s_low = stem.lower()
        a_low = alias.lower()
        return (s_low == a_low) or (a_low in s_low)

    for key, aliases in LOBAR_FILENAME_ALIASES.items():
        for alias in aliases:
            for stem, p in by_stem.items():
                if match_alias(stem, alias):
                    found[key] = p
                    break
            if key in found:
                break
    return found

def combine_binary_masks_to_labelmap(paths_by_lobe: Dict[str, Path], reference_img: Path) -> nib.Nifti1Image:
    ref = nib.load(str(reference_img))
    shape = ref.get_fdata().shape
    lab = np.zeros(shape, dtype=np.int16)

    paint_order = ["LLL", "LUL", "RLL", "RML", "RUL"]
    for key in paint_order:
        if key not in paths_by_lobe:
            continue
        arr = nib.load(str(paths_by_lobe[key])).get_fdata()
        mask = arr > 0.5
        lab[mask] = LABELS[key]
    return nib.Nifti1Image(lab, ref.affine, ref.header)

def ensure_output_lobemap(case_out_dir: Path, final_out_path: Path) -> None:
    single = find_single_lobemap(case_out_dir)
    if single is not None:
        shutil.copyfile(single, final_out_path)
        print(f"[OK] Copiado lobemap único: {single.name} -> {final_out_path.name}")
        return

    binaries = find_binary_masks(case_out_dir)
    if len(binaries) >= 2:
        ref_path = list(binaries.values())[0]
        combined = combine_binary_masks_to_labelmap(binaries, reference_img=ref_path)
        nib.save(combined, str(final_out_path))
        print(f"[OK] Combinadas {len(binaries)} máscaras -> {final_out_path.name}")
        return

    raise FileNotFoundError(f"No se encontró ni lobemap único ni suficientes máscaras binarias en {case_out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Batch lobe segmentation from NIfTI or DICOM.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images_dir", help="Carpeta con CTs NIfTI (.nii o .nii.gz).")
    group.add_argument("--dicom_root", help="Carpeta raíz con pacientes en DICOM (subcarpetas por paciente).")

    parser.add_argument("--out_dir", required=True,
                        help="Carpeta base de salida para los labelmaps por paciente.")
    parser.add_argument("--suffix_out", default="_from_dicom",
                        help="Sufijo que se añade a --out_dir cuando se usa --dicom_root (por defecto: _from_dicom).")
    parser.add_argument("--work_dir", default=None, help="Carpeta de trabajo intermedia por caso.")
    parser.add_argument("--fast", action="store_true", help="Usar modo --fast de TS.")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="Dispositivo para TS.")
    parser.add_argument("--overwrite", action="store_true", help="Rehacer casos ya presentes en out_dir.")
    parser.add_argument("--extra_ts_args", nargs=argparse.ONE_OR_MORE, default=None,
                        help="Args extra para TS. Ej: -ml")
    parser.add_argument("--min_series_files", type=int, default=30,
                        help="Mínimo de ficheros para considerar una serie válida (DICOM).")
    args = parser.parse_args()

    # Salida: si usamos DICOM, añade sufijo para diferenciar
    out_dir = Path(args.out_dir)
    if args.dicom_root:
        out_dir = out_dir.parent / f"{out_dir.name}{args.suffix_out}"
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir) if args.work_dir else (out_dir / "_ts_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.images_dir:
        # -------- MODO NIFTI (como antes)
        images_dir = Path(args.images_dir)
        image_paths = sorted([p for p in images_dir.iterdir()
                              if p.is_file() and p.name.endswith((".nii", ".nii.gz"))])
        if not image_paths:
            print(f"[ERROR] No se encontraron NIfTI en {images_dir}")
            sys.exit(1)

        for img in image_paths:
            pid = img.name.replace(".nii.gz", "").replace(".nii", "")
            final_out_path = out_dir / f"{pid}.nii.gz"
            if final_out_path.exists() and not args.overwrite:
                print(f"[SKIP] Ya existe: {final_out_path.name} (usa --overwrite)")
                continue

            case_work = work_dir / pid
            if case_work.exists():
                shutil.rmtree(case_work)
            case_work.mkdir(parents=True, exist_ok=True)

            ok, _ = run_totalsegmentator(
                image_path=img,
                case_out_dir=case_work,
                fast=args.fast,
                device=args.device,
                extra_args=args.extra_ts_args,
            )
            if not ok:
                print(f"[ERROR] Falló TS para {pid}")
                continue

            try:
                ensure_output_lobemap(case_work, final_out_path)
            except Exception as e:
                print(f"[ERROR] No se pudo generar lobemap final para {pid}: {e}", file=sys.stderr)
                continue

            try:
                shutil.rmtree(case_work)
            except Exception:
                pass

    else:
        # -------- MODO DICOM
        dicom_root = Path(args.dicom_root)
        patients = sorted([p for p in dicom_root.iterdir() if p.is_dir()])
        if not patients:
            print(f"[ERROR] No se encontraron subcarpetas de pacientes en {dicom_root}")
            sys.exit(1)

        for pdir in patients:
            pid = pdir.name
            final_out_path = out_dir / f"{pid}.nii.gz"
            if final_out_path.exists() and not args.overwrite:
                print(f"[SKIP] {pid}: ya existe (usa --overwrite)")
                continue

            series_dir = find_best_dicom_series(pdir, min_files=args.min_series_files)
            if series_dir is None:
                print(f"[ERROR] {pid}: no se encontró una serie DICOM válida (min_files={args.min_series_files})")
                continue

            case_work = work_dir / pid
            if case_work.exists():
                shutil.rmtree(case_work)
            case_work.mkdir(parents=True, exist_ok=True)

            # 1º intento: TS directamente sobre carpeta DICOM
            ok, stderr_text = run_totalsegmentator(
                image_path=series_dir,
                case_out_dir=case_work,
                fast=args.fast,
                device=args.device,
                extra_args=args.extra_ts_args,
            )

            # Si falla, 2º intento: convertir con dcm2niix -> NIfTI -> TS
            if not ok:
                print(f"[INFO] Reintentando {pid} vía dcm2niix...")
                nifti_dir = case_work / "dcm2niix"
                nifti_path = convert_dicom_to_nifti(series_dir, nifti_dir)
                if nifti_path is None:
                    print(f"[ERROR] {pid}: sin NIfTI para reintento. Caso fallido.")
                    continue

                ok, _ = run_totalsegmentator(
                    image_path=nifti_path,
                    case_out_dir=case_work,
                    fast=args.fast,
                    device=args.device,
                    extra_args=args.extra_ts_args,
                )
                if not ok:
                    print(f"[ERROR] {pid}: TS falló incluso tras convertir a NIfTI.")
                    continue

            # Generar lobemap final 1..5
            try:
                ensure_output_lobemap(case_work, final_out_path)
            except Exception as e:
                print(f"[ERROR] {pid}: no se pudo generar lobemap final: {e}", file=sys.stderr)
                continue

            # Limpieza
            try:
                shutil.rmtree(case_work)
            except Exception:
                pass

    print(f"[DONE] Segmentación de lóbulos completada. Resultados en: {out_dir}")

if __name__ == "__main__":
    main()
