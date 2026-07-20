import os
import numpy as np
import torch
import nibabel as nib
import traceback
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, TransposeD, ResizeD, ToTensord
)
from monai.data import Dataset as MonaiDictDataset

from transforms import Windowingd, MultiWindowingd  # tus transforms

# -----------------------
# Helpers
# -----------------------
def save_npy(path_wo_ext, array):
    os.makedirs(os.path.dirname(path_wo_ext), exist_ok=True)
    np.save(path_wo_ext + ".npy", array)

def save_nifti(path_wo_ext, array, affine):
    os.makedirs(os.path.dirname(path_wo_ext), exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), path_wo_ext + ".nii.gz")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# -----------------------
# Build items list (estructura {pid}_seg)
# -----------------------
def build_items_list_all(root_dir, patient_dirnames):
    items = []
    for dirname in patient_dirnames:
        if not dirname.endswith("_seg"):
            continue

        pid = dirname[:-4]  # quita "_seg"
        case_dir = os.path.join(root_dir, dirname)

        img  = os.path.join(case_dir, f"{pid}.nii.gz")
        lung = os.path.join(case_dir, "lung.nii.gz")
        nod  = os.path.join(case_dir, "lung_nodules.nii.gz")
        bro  = os.path.join(case_dir, "lung_trachea_bronchia.nii.gz")
        ves  = os.path.join(case_dir, "lung_vessels.nii.gz")

        if all(os.path.exists(p) for p in [img, lung, nod, bro, ves]):
            items.append({
                "image": img,
                "mask_lung": lung,
                "mask_nodule": nod,
                "mask_trachea_bronchia": bro,
                "mask_vessels": ves,
                "pid": pid
            })
    return items

# -----------------------
# Targets
# -----------------------
target_small   = (128, 256, 256)
target_medium  = (256, 512, 512)
target_cube64  = (64, 64, 64)
target_cube128 = (128, 128, 128)

# -----------------------
# Transforms (imagen + 4 máscaras)
# -----------------------
def make_transforms(spatial_size, add_windowing=None):
    """
    add_windowing:
      None
      ("single", center, width)
      ("multi", [(c,w),...])
    """
    keys_img = ["image"]
    keys_masks = ["mask_lung", "mask_nodule", "masks_trachea_bronchia", "mask_vessels"]
    keys_all = keys_img + keys_masks

    tfms = [
        LoadImaged(keys=keys_all),
        EnsureChannelFirstd(keys=keys_all),
        TransposeD(keys=keys_all, indices=(0, 3, 1, 2)),  # -> (C,D,H,W)
        ResizeD(keys=keys_img, spatial_size=spatial_size, mode="trilinear", align_corners=True),
        ResizeD(keys=keys_masks, spatial_size=spatial_size, mode="nearest"),
    ]

    # windowing solo a la imagen
    if add_windowing is not None:
        kind = add_windowing[0]
        if kind == "single":
            _, center, width = add_windowing
            tfms.append(Windowingd(keys=["image"], window_center=center, window_width=width))
        elif kind == "multi":
            _, windows = add_windowing
            tfms.append(MultiWindowingd(keys=["image"], windows=windows))

    tfms.append(ToTensord(keys=keys_all))
    return Compose(tfms)

# -----------------------
# Runner
# -----------------------
def run_preprocessing_configs(root_dir_cases, patient_dirnames, output_base_dir, configs):
    items = build_items_list_all(root_dir_cases, patient_dirnames)
    print(f"Encontrados {len(items)} casos válidos en {root_dir_cases}")

    for conf_name, conf in configs.items():
        print(f"\n>>> Config: {conf_name}")

        tfm = conf["transforms"]
        save_formats = conf.get("save_format", ["npy"])
        if isinstance(save_formats, str):
            save_formats = [save_formats]

        save_combined = conf.get("save_combined", False)  # lo mantienes por compatibilidad

        ds = MonaiDictDataset(data=items, transform=tfm)

        for i in tqdm(range(len(ds)), desc=f"[{conf_name}]"):
            try:
                sample = ds[i]
                pid = items[i]["pid"]

                img  = to_numpy(sample["image"])           # (C,D,H,W) si multiwindow
                lung = to_numpy(sample["mask_lung"])       # (1,D,H,W)
                nod  = to_numpy(sample["mask_nodule"])     # (1,D,H,W)
                bro  = to_numpy(sample["mask_trachea_bronchia"])   # (1,D,H,W)
                ves  = to_numpy(sample["mask_vessels"])    # (1,D,H,W)

                # affine
                meta = sample.get("image_meta_dict", None)
                aff = meta.get("affine", None) if meta is not None else None
                if aff is None:
                    aff = nib.load(items[i]["image"]).affine

                for fmt in save_formats:
                    base = os.path.join(output_base_dir, conf_name, fmt)

                    # paths salida
                    out_img  = os.path.join(base, "images", pid)
                    out_lung = os.path.join(base, "masks_lung", pid)
                    out_nod  = os.path.join(base, "masks_nodule", pid)
                    out_bro  = os.path.join(base, "masks_trachea_bronchia", pid)
                    out_ves  = os.path.join(base, "masks_vessels", pid)

                    if fmt == "npy":
                        save_npy(out_img,  img.squeeze())
                        save_npy(out_lung, lung.squeeze())
                        save_npy(out_nod,  nod.squeeze())
                        save_npy(out_bro,  bro.squeeze())
                        save_npy(out_ves,  ves.squeeze())
                    else:
                        save_nifti(out_img,  img.squeeze(), aff)
                        save_nifti(out_lung, lung.squeeze().astype(np.uint8), aff)
                        save_nifti(out_nod,  nod.squeeze().astype(np.uint8),  aff)
                        save_nifti(out_bro,  bro.squeeze().astype(np.uint8),  aff)
                        save_nifti(out_ves,  ves.squeeze().astype(np.uint8),  aff)

                    if save_combined:
                        # combined = [img_channels, lung, nodule, bronchia, vessels]
                        combined = np.concatenate([img, lung, nod, bro, ves], axis=0).astype(np.float32)
                        out_comb = os.path.join(base, "combined", pid)
                        if fmt == "npy":
                            save_npy(out_comb, combined)
                        else:
                            save_nifti(out_comb, combined, aff)

            except Exception:
                print(f"Error en {items[i].get('pid','?')}")
                traceback.print_exc()

# -----------------------
# Configs (todas las tuyas)
# -----------------------
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
        "save_combined": False,
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

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    root_cases = "/mnt/homeGPU/mcribilles/tfm/segmentation/segmentaciones_lung_and_nodules_vessels_bronquia/nuevos17dic2025/"
    patient_dirnames = sorted(os.listdir(root_cases))

    out_base = "/mnt/homeGPU/mcribilles/tfm/volumenes_preprocesados/"
    run_preprocessing_configs(root_cases, patient_dirnames, out_base, configs)
