import json

OLD = "./radiomica/valid_configurations.json"                  # ruta al viejo
NEW = "./radiomica/valid_configurations_radiomics.json"        # salida

# Ajusta estas rutas si tus archivos están en otro sitio:
FEATURES_CSV = "./data/radiomics_features_lung_nodule.csv"
LABELS_CSV   = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data.csv"

COMMON_KEYS = {
    "features_csv": FEATURES_CSV,
    "labels_csv": LABELS_CSV,
    "id_col": "Id_paciente",
    "label_col": "Complicación",
    "label_positive": "Sí",
    "use_masks": "both_merge",   # "lung" o "nodule" si quieres forzar
    "n_splits": 5,
    "random_state": 42,
    "shuffle_folds": True
}

with open(OLD, "r") as f:
    old_cfgs = json.load(f)

new_cfgs = []
for cfg in old_cfgs:
    cfg2 = {k: v for k, v in cfg.items() if k != "dataset_path"}  # quita dataset_path
    # añade los campos nuevos solo si no existen ya
    for k, v in COMMON_KEYS.items():
        cfg2.setdefault(k, v)
    new_cfgs.append(cfg2)

with open(NEW, "w") as f:
    json.dump(new_cfgs, f, indent=2, ensure_ascii=False)

print(f" Convertido: {len(new_cfgs)} configuraciones → {NEW}")
