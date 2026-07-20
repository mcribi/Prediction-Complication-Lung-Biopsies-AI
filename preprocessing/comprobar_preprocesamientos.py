import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm


MODALITIES = [
    "images",
    "masks_lung",
    "masks_nodule",
    "masks_trachea_bronchia",
    "masks_vessels",
]

MASK_MODALITIES = {
    "masks_lung",
    "masks_nodule",
    "masks_trachea_bronchia",
    "masks_vessels",
}

OVERLAP_TARGETS = {
    "masks_nodule": 0.5,
    "masks_trachea_bronchia": 0.5,
    "masks_vessels": 0.5,
}

CSV_COLUMNS = [
    "root",
    "config",
    "format",
    "pid",
    "modality",
    "file_path",
    "check",
    "status",   # PASS, FAIL, WARN, INFO, SKIP
    "value",
    "details",
]


def is_config_dir(name: str) -> bool:
    return name.startswith("resize_") and not name.startswith(".")


def is_windowed_config(conf_name: str) -> bool:
    # Heurística: configs con windowing o multiwindowing
    return ("_hu_" in conf_name) or ("multiwindowing" in conf_name)


def list_configs(root_dir: str) -> List[str]:
    return sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and is_config_dir(d)
    ])


def parse_pid_from_filename(fn: str, fmt: str) -> Optional[str]:
    if fmt == "npy":
        if fn.endswith(".npy"):
            return fn[:-4]
        return None
    # nifti
    if fn.endswith(".nii.gz"):
        return fn[:-7]
    return None


def list_pid_to_file(mod_dir: str, fmt: str) -> Dict[str, str]:
    mapping = {}
    if not os.path.isdir(mod_dir):
        return mapping
    for fn in os.listdir(mod_dir):
        pid = parse_pid_from_filename(fn, fmt)
        if pid is None:
            continue
        mapping[pid] = os.path.join(mod_dir, fn)
    return mapping


def load_array(path: str, fmt: str) -> np.ndarray:
    if fmt == "nifti":
        return np.asanyarray(nib.load(path).dataobj)
    return np.load(path)


def get_shape_fast(path: str, fmt: str) -> Tuple[Tuple[int, ...], str]:
    if fmt == "nifti":
        img = nib.load(path)  # lazy
        return tuple(img.shape), str(img.get_data_dtype())
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape), str(arr.dtype)


def bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    zmin, ymin, xmin = idx.min(axis=0)
    zmax, ymax, xmax = idx.max(axis=0)
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def centroid(mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    c = idx.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])


def touches_border(bb: Optional[Tuple[int, int, int, int, int, int]], shape_3d: Tuple[int, int, int], margin: int = 0) -> bool:
    if bb is None:
        return False
    D, H, W = shape_3d
    zmin, zmax, ymin, ymax, xmin, xmax = bb
    return (
        zmin <= margin or ymin <= margin or xmin <= margin or
        zmax >= (D - 1 - margin) or ymax >= (H - 1 - margin) or xmax >= (W - 1 - margin)
    )


def mask_binary_check(mask: np.ndarray, max_unique: int = 10) -> Tuple[bool, List[float]]:
    # Para máscaras grandes, unique completo puede ser caro, pero normalmente aquí van uint8/bool y es rápido.
    u = np.unique(mask)
    if u.size > max_unique:
        u = u[:max_unique]
    ok = set(u.tolist()).issubset({0, 1})
    return ok, u.tolist()


def overlap_ratio(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = (a > 0)
    b = (b > 0)
    va = int(a.sum())
    if va == 0:
        return None
    inter = int(np.logical_and(a, b).sum())
    return float(inter) / float(va)


def image_stats(arr: np.ndarray) -> Dict[str, float]:
    x = arr.astype(np.float32, copy=False)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p1": float(np.percentile(x, 1)),
        "p99": float(np.percentile(x, 99)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def masked_stats(img: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    m = (mask > 0)
    if int(m.sum()) == 0:
        return None
    x = img[m].astype(np.float32, copy=False)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p5": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "nvox": int(x.size),
    }


def write_row(rows: List[dict], **kwargs):
    row = {k: "" for k in CSV_COLUMNS}
    for k, v in kwargs.items():
        if k in row:
            row[k] = v
    rows.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raíz: .../volumenes_preprocesados/nuevos17dic2025")
    ap.add_argument("--formats", default="npy,nifti", help="npy,nifti o solo npy")
    ap.add_argument("--output", default="preproc_qc_report.csv", help="CSV de salida (se guarda donde ejecutes)")
    ap.add_argument("--max_patients", type=int, default=0, help="0 = todos, si no, limita pacientes por config")
    ap.add_argument("--overlap_thr", type=float, default=0.5, help="Umbral overlap con pulmón para nodule/bronchia/vessels")
    args = ap.parse_args()

    root_dir = args.root
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    configs = list_configs(root_dir)

    rows: List[dict] = []

    for conf in configs:
        conf_path = os.path.join(root_dir, conf)
        conf_windowed = is_windowed_config(conf)

        for fmt in formats:
            fmt_path = os.path.join(conf_path, fmt)
            if not os.path.isdir(fmt_path):
                write_row(
                    rows,
                    root=root_dir, config=conf, format=fmt, pid="",
                    modality="", file_path=fmt_path,
                    check="format_dir_exists", status="FAIL",
                    value="missing", details="No existe carpeta del formato."
                )
                continue

            # Mapas pid -> file por modalidad
            maps = {}
            all_pids = set()

            for mod in MODALITIES:
                mod_dir = os.path.join(fmt_path, mod)
                if not os.path.isdir(mod_dir):
                    write_row(
                        rows,
                        root=root_dir, config=conf, format=fmt, pid="",
                        modality=mod, file_path=mod_dir,
                        check="modality_dir_exists", status="FAIL",
                        value="missing", details="No existe carpeta de modalidad."
                    )
                    maps[mod] = {}
                    continue

                maps[mod] = list_pid_to_file(mod_dir, fmt)
                all_pids |= set(maps[mod].keys())

            pids = sorted(list(all_pids))
            if args.max_patients and args.max_patients > 0:
                pids = pids[:args.max_patients]

            for pid in tqdm(pids, desc=f"[{conf} | {fmt}]"):
                # Presencia de archivos por modalidad
                paths = {mod: maps.get(mod, {}).get(pid) for mod in MODALITIES}

                for mod in MODALITIES:
                    if paths[mod] is None:
                        write_row(
                            rows,
                            root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path="",
                            check="file_exists", status="FAIL",
                            value="missing", details="Falta el archivo para este pid y modalidad."
                        )
                    else:
                        write_row(
                            rows,
                            root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=paths[mod],
                            check="file_exists", status="PASS",
                            value="present", details=""
                        )

                # Si no hay imagen, no seguimos con checks dependientes
                img_path = paths["images"]
                if img_path is None:
                    write_row(
                        rows, root=root_dir, config=conf, format=fmt, pid=pid,
                        modality="images", file_path="",
                        check="image_dependent_checks", status="SKIP",
                        value="", details="Sin imagen, no se pueden validar shapes, stats ni alineación."
                    )
                    continue

                # Shapes rápidos
                img_shape, img_dtype = get_shape_fast(img_path, fmt)
                write_row(
                    rows, root=root_dir, config=conf, format=fmt, pid=pid,
                    modality="images", file_path=img_path,
                    check="shape", status="INFO",
                    value=str(img_shape), details=f"dtype={img_dtype}"
                )

                # Carga imagen para stats y checks
                try:
                    img = load_array(img_path, fmt)
                except Exception as e:
                    write_row(
                        rows, root=root_dir, config=conf, format=fmt, pid=pid,
                        modality="images", file_path=img_path,
                        check="load", status="FAIL",
                        value="", details=repr(e)
                    )
                    continue

                # Imagen: finitos
                finite_ok = bool(np.isfinite(img).all())
                write_row(
                    rows, root=root_dir, config=conf, format=fmt, pid=pid,
                    modality="images", file_path=img_path,
                    check="finite", status=("PASS" if finite_ok else "FAIL"),
                    value=str(finite_ok), details=("OK" if finite_ok else "Contiene NaN o Inf")
                )

                # Imagen: stats por canal (si multiwindow)
                if img.ndim == 4:
                    # (C,D,H,W)
                    for c in range(img.shape[0]):
                        st = image_stats(img[c])
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=f"images_c{c}", file_path=img_path,
                            check="range_stats", status="INFO",
                            value=f"min={st['min']:.2f},max={st['max']:.2f},p1={st['p1']:.2f},p99={st['p99']:.2f}",
                            details=f"mean={st['mean']:.2f},std={st['std']:.2f}"
                        )
                    img0 = img[0]
                else:
                    st = image_stats(img)
                    write_row(
                        rows, root=root_dir, config=conf, format=fmt, pid=pid,
                        modality="images", file_path=img_path,
                        check="range_stats", status="INFO",
                        value=f"min={st['min']:.2f},max={st['max']:.2f},p1={st['p1']:.2f},p99={st['p99']:.2f}",
                        details=f"mean={st['mean']:.2f},std={st['std']:.2f}"
                    )
                    img0 = img

                # Cargar lung si existe (necesario para overlap y centroid dist)
                lung_path = paths["masks_lung"]
                lung = None
                if lung_path is not None:
                    try:
                        lung = load_array(lung_path, fmt)
                    except Exception as e:
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality="masks_lung", file_path=lung_path,
                            check="load", status="FAIL",
                            value="", details=repr(e)
                        )
                        lung = None

                # Checks por cada máscara
                for mod in MASK_MODALITIES:
                    mpath = paths.get(mod)
                    if mpath is None:
                        continue

                    # shape fast
                    try:
                        mshape, mdtype = get_shape_fast(mpath, fmt)
                        # shape mismatch check: comparar con forma "espacial"
                        # imagen puede ser 3D o 4D. Si es 4D, la parte espacial es (D,H,W)
                        img_spatial = img0.shape
                        ok_shape = (mshape == img_spatial) or (mshape == (1,) + img_spatial)

                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="shape_match_image", status=("PASS" if ok_shape else "FAIL"),
                            value=str(mshape), details=f"img_spatial={img_spatial}, dtype={mdtype}"
                        )
                    except Exception as e:
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="shape_match_image", status="FAIL",
                            value="", details=f"Error leyendo shape: {repr(e)}"
                        )
                        continue

                    # cargar máscara
                    try:
                        mask = load_array(mpath, fmt)
                        # normalizar a 3D si viene como (1,D,H,W)
                        if mask.ndim == 4 and mask.shape[0] == 1:
                            mask3 = mask[0]
                        else:
                            mask3 = mask
                    except Exception as e:
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="load", status="FAIL",
                            value="", details=repr(e)
                        )
                        continue

                    # binaria
                    ok_bin, uniq = mask_binary_check(mask3)
                    write_row(
                        rows, root=root_dir, config=conf, format=fmt, pid=pid,
                        modality=mod, file_path=mpath,
                        check="binary", status=("PASS" if ok_bin else "FAIL"),
                        value=str(ok_bin), details=f"unique={uniq}"
                    )

                    # bbox y borde
                    bb = bbox(mask3)
                    write_row(
                        rows, root=root_dir, config=conf, format=fmt, pid=pid,
                        modality=mod, file_path=mpath,
                        check="bbox", status="INFO",
                        value=str(bb), details=""
                    )
                    if bb is not None:
                        tb = touches_border(bb, img0.shape, margin=0)
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="touches_border", status=("WARN" if tb else "PASS"),
                            value=str(tb), details="WARN si toca borde"
                        )
                    else:
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="touches_border", status="INFO",
                            value="empty", details="Máscara vacía."
                        )

                    # centroid
                    c = centroid(mask3)
                    write_row(
                        rows, root=root_dir, config=conf, format=fmt, pid=pid,
                        modality=mod, file_path=mpath,
                        check="centroid", status="INFO",
                        value=str(c), details=""
                    )

                    # overlap con lung para masks no-lung
                    if mod != "masks_lung":
                        if lung is None:
                            write_row(
                                rows, root=root_dir, config=conf, format=fmt, pid=pid,
                                modality=mod, file_path=mpath,
                                check="overlap_with_lung", status="SKIP",
                                value="", details="Sin lung disponible."
                            )
                        else:
                            # lung a 3D
                            lung3 = lung[0] if (lung.ndim == 4 and lung.shape[0] == 1) else lung
                            ov = overlap_ratio(mask3, lung3)
                            if ov is None:
                                write_row(
                                    rows, root=root_dir, config=conf, format=fmt, pid=pid,
                                    modality=mod, file_path=mpath,
                                    check="overlap_with_lung", status="INFO",
                                    value="empty", details="Máscara vacía."
                                )
                            else:
                                thr = args.overlap_thr
                                ok_ov = (ov >= thr)
                                write_row(
                                    rows, root=root_dir, config=conf, format=fmt, pid=pid,
                                    modality=mod, file_path=mpath,
                                    check="overlap_with_lung", status=("PASS" if ok_ov else "FAIL"),
                                    value=f"{ov:.4f}", details=f"threshold={thr}"
                                )

                    # stats HU dentro de máscara (usando img0)
                    st_in = masked_stats(img0, mask3)
                    if st_in is None:
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="hu_stats_inside_mask", status="INFO",
                            value="empty", details="Máscara vacía."
                        )
                    else:
                        # El significado de HU depende del windowing
                        status = "INFO" if conf_windowed else "INFO"
                        write_row(
                            rows, root=root_dir, config=conf, format=fmt, pid=pid,
                            modality=mod, file_path=mpath,
                            check="hu_stats_inside_mask", status=status,
                            value=f"mean={st_in['mean']:.2f},std={st_in['std']:.2f}",
                            details=f"p5={st_in['p5']:.2f},p95={st_in['p95']:.2f},nvox={st_in['nvox']}"
                        )

                # También stats HU en lung, por si hay cosas raras
                if lung is not None:
                    lung3 = lung[0] if (lung.ndim == 4 and lung.shape[0] == 1) else lung
                    st_l = masked_stats(img0, lung3)
                    if st_l is not None:
                        # Heurística suave solo como WARN si no es windowed
                        if (not conf_windowed) and (st_l["mean"] > -200):
                            write_row(
                                rows, root=root_dir, config=conf, format=fmt, pid=pid,
                                modality="masks_lung", file_path=paths["masks_lung"],
                                check="lung_mean_sanity", status="WARN",
                                value=f"{st_l['mean']:.2f}",
                                details="Media de HU en pulmón muy alta para HU crudo. Revisar posible desalineación o intensidad no-HU."
                            )
                        else:
                            write_row(
                                rows, root=root_dir, config=conf, format=fmt, pid=pid,
                                modality="masks_lung", file_path=paths["masks_lung"],
                                check="lung_mean_sanity", status="PASS",
                                value=f"{st_l['mean']:.2f}",
                                details="OK (o config windowed)."
                            )

    # Guardar CSV en el directorio desde donde ejecutas
    out_csv = os.path.abspath(args.output)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nCSV guardado en: {out_csv}")
    print(f"Filas totales: {len(rows)}")
    print("Consejo: filtra status=FAIL o WARN para ver problemas rápido.")


if __name__ == "__main__":
    main()
