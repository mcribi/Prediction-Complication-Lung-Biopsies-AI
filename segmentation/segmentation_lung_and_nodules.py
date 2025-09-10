from totalsegmentator.python_api import totalsegmentator
import os
from tqdm import tqdm

def segmentation_lung_and_nodules(input_dir, output_dir, force_cpu=False):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in tqdm(os.listdir(input_dir)):
        if not (archivo.endswith(".nii") or archivo.endswith(".nii.gz")):
            continue

        ruta_entrada = os.path.join(input_dir, archivo)
        # TotalSegmentator crea una carpeta y dentro guarda los .nii.gz por etiqueta
        ruta_salida = os.path.join(
            output_dir, archivo.replace(".nii.gz","").replace(".nii","") + "_seg"
        )

        print(f"Segmentando (task=lung_nodules): {archivo}")
        totalsegmentator(
            input=ruta_entrada,
            output=ruta_salida,
            task="lung_nodules",   # <- cambio clave
            ml=True,
            fast=False,            # prueba sin fast
            device="cpu" if force_cpu else "gpu",
            verbose=True
        )
        print(f"Hecho: {ruta_salida}")


if __name__ == "__main__":
    input_dir = "/mnt/homeGPU/mcribilles/TFG/nifti_convertidos_anonimizados"
    output_dir = "./segmentaciones_lung"
    segmentation_lung_and_nodules(input_dir, output_dir, force_cpu=False)