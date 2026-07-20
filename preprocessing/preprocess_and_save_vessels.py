import os, numpy as np, torch, nibabel as nib, traceback
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, TransposeD, ResizeD,
    ToTensord, MapTransform
)
from transforms import Windowingd, MultiWindowingd, DebugShaped  

#helpers for saving
def save_npy(path_wo_ext, array):
    os.makedirs(os.path.dirname(path_wo_ext), exist_ok=True)
    np.save(path_wo_ext + ".npy", array)

def save_nifti(path_wo_ext, array, affine):
    os.makedirs(os.path.dirname(path_wo_ext), exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), path_wo_ext + ".nii.gz")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)

# dataset monai triplete
def build_items_list_triplet(root_dir, patient_dirnames):
    items = []
    for dirname in patient_dirnames:
        if not dirname.endswith("_seg"):
            continue
        pid = dirname[:-4]  # quita "_seg"
        case_dir = os.path.join(root_dir, dirname)
        img  = os.path.join(case_dir, f"{pid}.nii.gz")
        lung = os.path.join(case_dir, "lung.nii.gz")
        nod  = os.path.join(case_dir, "lung_nodules.nii.gz")
        if all(os.path.exists(p) for p in [img, lung, nod]):
            items.append({"image": img, "mask_lung": lung, "mask_nodule": nod, "pid": pid})
    return items

def build_items_list_vessels(root_dir, patient_dirnames):
    """
    Espera carpetas del tipo {pid}_vessels con:
      lung_trachea_bronchia.nii.gz
      lung_vessels.nii.gz
    """
    items = []
    for dirname in patient_dirnames:
        if not dirname.endswith("_vessels"):
            continue
        pid = dirname[:-8]  # quita "_vessels"
        case_dir = os.path.join(root_dir, dirname)
        trachea = os.path.join(case_dir, "lung_trachea_bronchia.nii.gz")
        vessels = os.path.join(case_dir, "lung_vessels.nii.gz")
        if all(os.path.exists(p) for p in [trachea, vessels]):
            items.append({
                "mask_trachea": trachea,
                "mask_vessels": vessels,
                "pid": pid
            })
    return items



def make_vessels_transforms(spatial_size):
    """
    Mismas operaciones geométricas que el resto (Transpose, Resize)
    pero solo para máscaras de tráquea/bronquios y vasos.
    """
    return Compose([
        LoadImaged(keys=["mask_trachea", "mask_vessels"]),
        EnsureChannelFirstd(keys=["mask_trachea", "mask_vessels"]),
        TransposeD(keys=["mask_trachea", "mask_vessels"], indices=(0, 3, 1, 2)),  # -> (1,D,H,W)
        ResizeD(keys=["mask_trachea", "mask_vessels"], spatial_size=spatial_size, mode="nearest"),
        ToTensord(keys=["mask_trachea", "mask_vessels"]),
    ])


def spatial_size_from_conf_name(conf_name):
    """
    Usamos las mismas reglas que tus configs:
      - "small"   -> target_small
      - "medium"  -> target_medium
      - "cube64"  -> target_cube64
      - "cube128" -> target_cube128
    """
    if "cube128" in conf_name:
        return target_cube128
    if "cube64" in conf_name:
        return target_cube64
    if "medium" in conf_name:
        return target_medium
    # por defecto "small"
    return target_small



def run_preprocessing_configs_vessels(root_dir_cases, patient_ids, output_base_dir, configs_base):
    """
    Recorre las MISMAS configs que el triplete (mismos nombres de carpeta)
    pero usando transforms específicos de vessels.
    """
    items = build_items_list_vessels(root_dir_cases, patient_ids)
    print(f"Encontrados {len(items)} casos de vessels válidos en {root_dir_cases}")

    for conf_name, conf in configs_base.items():
        print(f"\n>>> Config (vessels): {conf_name}")
        spatial_size = spatial_size_from_conf_name(conf_name)
        tfm = make_vessels_transforms(spatial_size)

        save_formats = conf.get("save_format", ["npy"])
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        ds = MonaiDictDataset(data=items, transform=tfm)

        for i in tqdm(range(len(ds)), desc=f"[vessels_{conf_name}]"):
            try:
                sample = ds[i]
                pid = items[i]["pid"]

                trachea = to_numpy(sample["mask_trachea"])   # (1,D,H,W)
                vessels = to_numpy(sample["mask_vessels"])   # (1,D,H,W)

                for fmt in save_formats:
                    base = os.path.join(output_base_dir, conf_name, fmt)

                    # affine desde la mask de tráquea (como referencia)
                    meta = sample.get("mask_trachea_meta_dict", None)
                    aff = None
                    if meta is not None:
                        aff = meta.get("affine", None)
                    if aff is None:
                        aff = nib.load(items[i]["mask_trachea"]).affine

                    out_tr = os.path.join(base, "masks_trachea_bronchia", pid)
                    out_vs = os.path.join(base, "masks_vessels", pid)

                    if fmt == "npy":
                        save_npy(out_tr, trachea.squeeze())
                        save_npy(out_vs, vessels.squeeze())
                    else:
                        save_nifti(out_tr, trachea.squeeze().astype(np.uint8), aff)
                        save_nifti(out_vs, vessels.squeeze().astype(np.uint8), aff)

            except Exception:
                print(f" Error en {items[i]['pid']} (vessels)")
                traceback.print_exc()





from monai.data import Dataset as MonaiDictDataset

# Transforms base (3 canales: imagen, pulmón, nódulo)
target_small  = (128, 256, 256)
target_medium = (256, 512, 512)
target_cube64  = (64, 64, 64)
target_cube128 = (128, 128, 128)

def make_transforms(spatial_size, add_windowing=None):
    """
    add_windowing: None | ("single", center, width) | ("multi", [(c,w),...])
    """
    tfms = [
        LoadImaged(keys=["image","mask_lung","mask_nodule"]),
        EnsureChannelFirstd(keys=["image","mask_lung","mask_nodule"]),
        TransposeD(keys=["image","mask_lung","mask_nodule"], indices=(0,3,1,2)),  # -> (1,D,H,W)
        ResizeD(keys=["image"], spatial_size=spatial_size, mode="trilinear", align_corners=True),
        ResizeD(keys=["mask_lung","mask_nodule"], spatial_size=spatial_size, mode="nearest"),
    ]
    # ventana HU a "image"
    if add_windowing is not None:
        kind = add_windowing[0]
        if kind == "single":
            _, center, width = add_windowing
            from transforms import Windowingd
            tfms.append(Windowingd(keys=["image"], window_center=center, window_width=width))
        elif kind == "multi":
            _, windows = add_windowing
            from transforms import MultiWindowingd
            tfms.append(MultiWindowingd(keys=["image"], windows=windows))
    tfms.append(ToTensord(keys=["image","mask_lung","mask_nodule"]))
    return Compose(tfms)

# runner
def run_preprocessing_configs_triplet(root_dir_cases, patient_ids, output_base_dir, configs):
    items = build_items_list_triplet(root_dir_cases, patient_ids)
    print(f"Encontrados {len(items)} casos válidos en {root_dir_cases}")

    for conf_name, conf in configs.items():
        print(f"\n>>> Config: {conf_name}")
        tfm = conf["transforms"]
        save_formats = conf.get("save_format", ["npy"])
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        save_combined = conf.get("save_combined", False)  # exportar tensor apilado (3 canales)

        ds = MonaiDictDataset(data=items, transform=tfm)

        for i in tqdm(range(len(ds)), desc=f"[{conf_name}]"):
            try:
                sample = ds[i]
                pid = items[i]["pid"]
                img = to_numpy(sample["image"])           # (1,D,H,W) o multicanal (multiwindow)
                lung= to_numpy(sample["mask_lung"])       # (1,D,H,W) binaria
                nod = to_numpy(sample["mask_nodule"])     # (1,D,H,W) binaria

                # En caso de single-channel image, nos quedamos con img[0]
                # Si aplicaste MultiWindowingd tendrás img con C>1; para guardar "images" en 1 canal, usa canal 0
                img_to_save = img if img.shape[0] > 1 else img[0:1]

                for fmt in save_formats:
                    base = os.path.join(output_base_dir, conf_name, fmt)

                    aff = None
                    meta = sample.get("image_meta_dict", None)
                    if meta is not None:
                        aff = meta.get("affine", None)

                    if aff is None:
                        # fallback robusto: leer el affine del fichero de entrada
                        aff = nib.load(items[i]["image"]).affine

                    # imágenes
                    out_img = os.path.join(base, "images", pid)
                    if fmt == "npy":  save_npy(out_img, img_to_save.squeeze())
                    else:             save_nifti(out_img, img_to_save.squeeze(), aff)

                    # máscaras
                    out_lung = os.path.join(base, "masks_lung", pid)
                    out_nod  = os.path.join(base, "masks_nodule", pid)
                    if fmt == "npy":
                        save_npy(out_lung, lung.squeeze()); save_npy(out_nod, nod.squeeze())
                    else:
                        save_nifti(out_lung, lung.squeeze().astype(np.uint8), aff)
                        save_nifti(out_nod,  nod.squeeze().astype(np.uint8),  aff)

                    # combinado
                    if save_combined:
                        combined = np.concatenate([img_to_save, lung, nod], axis=0).astype(np.float32)
                        out_comb = os.path.join(base, "combined", pid)
                        if fmt == "npy": save_npy(out_comb, combined)
                        else:            save_nifti(out_comb, combined, aff)


            except Exception:
                print(f" Error en {items[i]['pid']}")
                traceback.print_exc()

# configs
configs = {
    "resize_small": {
        "transforms": make_transforms(target_small),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
    "resize_small_hu_m600_1500": {
        "transforms": make_transforms(target_small, add_windowing=("single",-600,1500)),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
    "resize_small_hu_m300_1400": {
        "transforms": make_transforms(target_small, add_windowing=("single",-300,1400)),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
    "resize_small_multiwindowing_separadas": {
        "transforms": make_transforms(target_small, add_windowing=("multi",[(-600,1500),(40,400),(-160,600)])),
        "save_format": ["npy","nifti"],
        "save_combined": False,  #True si quieres (C=3) en combined
    },
    "resize_medium": {
        "transforms": make_transforms(target_medium),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
    "resize_medium_hu_m600_1500": {
        "transforms": make_transforms(target_medium, add_windowing=("single",-600,1500)),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
    "resize_medium_hu_m300_1400": {
        "transforms": make_transforms(target_medium, add_windowing=("single",-300,1400)),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
    "resize_medium_multiwindowing_separadas": {
        "transforms": make_transforms(target_medium, add_windowing=("multi",[(-600,1500),(40,400),(-160,600)])),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },

    "resize_cube64": {
        "transforms": make_transforms(target_cube64),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },

    "resize_cube64_hu_m600_1500": {
    "transforms": make_transforms(target_cube64, add_windowing=("single",-600,1500)),
    "save_format": ["npy","nifti"],
    "save_combined": False,
    },

    "resize_cube64_hu_m300_1400": {
    "transforms": make_transforms(target_cube64, add_windowing=("single",-300,1400)),
    "save_format": ["npy","nifti"],
    "save_combined": False,
    },

    "resize_cube64_multiwindowing_separadas": {
        "transforms": make_transforms(target_cube64, add_windowing=("multi",[(-600,1500),(40,400),(-160,600)])),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },

    "resize_cube128": {
        "transforms": make_transforms(target_cube128),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },

    "resize_cube128_hu_m600_1500": {
    "transforms": make_transforms(target_cube128, add_windowing=("single",-600,1500)),
    "save_format": ["npy","nifti"],
    "save_combined": False,
    },


    "resize_cube128_hu_m300_1400": {
    "transforms": make_transforms(target_cube128, add_windowing=("single",-300,1400)),
    "save_format": ["npy","nifti"],
    "save_combined": False,
    },

    "resize_cube128_multiwindowing_separadas": {
        "transforms": make_transforms(target_cube128, add_windowing=("multi",[(-600,1500),(40,400),(-160,600)])),
        "save_format": ["npy","nifti"],
        "save_combined": False,
    },
}

# main
if __name__ == "__main__":
    out_base = "./../volumenes_preprocesados/"

    # Solo queremos procesar vessels y tráquea, las imágenes ya están hechas
    root_cases_vessels = "/mnt/homeGPU/mcribilles/tfm/segmentation/segmentacion_vessels/"
    patient_dirnames_v = sorted(os.listdir(root_cases_vessels))

    # Si quieres usar TODAS las configs que ya tenías:
    configs_for_vessels = configs

    # Si solo quieres, por ejemplo, resize_small_hu_m300_1400:
    # configs_for_vessels = {
    #     k: v for k, v in configs.items()
    #     if k in ["resize_small_hu_m300_1400"]
    # }

    run_preprocessing_configs_vessels(
        root_dir_cases=root_cases_vessels,
        patient_ids=patient_dirnames_v,
        output_base_dir=out_base,
        configs_base=configs_for_vessels
    )
