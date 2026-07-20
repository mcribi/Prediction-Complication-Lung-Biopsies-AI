from totalsegmentator.python_api import totalsegmentator
import os
from tqdm import tqdm

def segmentacion_vessels(input_dir, output_dir, force_cpu=False, fast=False):
    """
    Segmenta vasos pulmonares (lung_vessels) en todos los volúmenes NIfTI del directorio.

    Args:
        input_dir (str): Carpeta con los volúmenes .nii o .nii.gz originales.
        output_dir (str): Carpeta donde se guardarán las segmentaciones.
        force_cpu (bool): Si True, fuerza uso de CPU (por defecto usa GPU si hay disponible).
        fast (bool): Usa el modo rápido de TotalSegmentator (menor resolución, más rápido).
    """
    os.makedirs(output_dir, exist_ok=True)

    for archivo in tqdm(os.listdir(input_dir)):
        if not (archivo.endswith(".nii") or archivo.endswith(".nii.gz")):
            continue

        ruta_entrada = os.path.join(input_dir, archivo)
        nombre_base = archivo.replace(".nii.gz", "").replace(".nii", "")
        ruta_salida = os.path.join(output_dir, f"{nombre_base}_vessels")

        print(f" Segmentando vasos pulmonares: {archivo}")
        try:
            totalsegmentator(
                input=ruta_entrada,
                output=ruta_salida,
                task="lung_vessels",       # Subtarea específica de vasos pulmonares
                fast=fast,
                device="cpu" if force_cpu else "gpu",
                verbose=True
            )
            print(f" Hecho: {ruta_salida}")
        except Exception as e:
            print(f" Error segmentando {archivo}: {e}")

    print("\n Segmentación de vasos completada.")


if __name__ == "__main__":
    input_dir = "/mnt/homeGPU/mcribilles/TFG/volumenes/nifti_convertidos_anonimizados/nuevos17dic25/"
    output_dir = "../segmentation/segmentacion_vessels/nuevos17dic25/"

    segmentacion_vessels(input_dir, output_dir, force_cpu=False, fast=False)
