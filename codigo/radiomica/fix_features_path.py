import json
import os

# rutas
CONFIG_FILE = "/mnt/homeGPU/mcribilles/tfm/codigo/radiomica/valid_configurations_radiomics.json"
OUTPUT_FILE = "/mnt/homeGPU/mcribilles/tfm/codigo/radiomica/valid_configurations_radiomics_fixed.json"

NEW_PATH = "/mnt/homeGPU/mcribilles/tfm/codigo/data/radiomic_data/radiomics_lung_nodule.csv"

# cargar
with open(CONFIG_FILE, "r") as f:
    configs = json.load(f)

# cambiar solo features_csv
for cfg in configs:
    if "features_csv" in cfg:
        cfg["features_csv"] = NEW_PATH

# guardar a fichero nuevo (para no machacar el original)
with open(OUTPUT_FILE, "w") as f:
    json.dump(configs, f, indent=2, ensure_ascii=True)

print("Guardado en {}".format(OUTPUT_FILE))
