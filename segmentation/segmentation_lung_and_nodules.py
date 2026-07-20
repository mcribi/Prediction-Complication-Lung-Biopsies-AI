from totalsegmentator.python_api import totalsegmentator
import os
from tqdm import tqdm

def segmentation_lung_and_nodules(input_dir, output_dir, force_cpu=False):
    os.makedirs(output_dir, exist_ok=True)

    for archivo in tqdm(os.listdir(input_dir)):
        if not (archivo.endswith(".nii") or archivo.endswith(".nii.gz")):
            continue

        ruta_entrada = os.path.join(input_dir, archivo)
        #TotalSegmentator create a directory and it save in the .nii.gz by target
        ruta_salida = os.path.join(
            output_dir, archivo.replace(".nii.gz","").replace(".nii","") + "_seg"
        )

        print(f"Segmentando (task=lung_nodules): {archivo}")
        totalsegmentator(
            input=ruta_entrada,
            output=ruta_salida,
            task="lung_nodules",   #instead of total, we now segment lung nodules
            #ml=True, #the two mask (nodules and lung) are in the same file
            fast=False,       
            device="cpu" if force_cpu else "gpu",
            verbose=True
        )
        print(f"Hecho: {ruta_salida}")


if __name__ == "__main__":
    input_dir = "/mnt/homeGPU/mcribilles/TFG/volumenes/nifti_convertidos_anonimizados/nuevos17dic25/"
    output_dir = "/mnt/homeGPU/mcribilles/TFG/segmentacion/segmentaciones_nodulos/nuevos17dic25/"
    segmentation_lung_and_nodules(input_dir, output_dir, force_cpu=False)