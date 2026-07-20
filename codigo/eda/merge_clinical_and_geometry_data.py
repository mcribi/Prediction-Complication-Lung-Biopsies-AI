import pandas as pd

CLINICAL_SD_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data_limpios_SD.csv"
GEOM_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/nodule_geometry_210.csv"
OUT_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data_SD_con_geometria_210.csv"

df_cli = pd.read_csv(CLINICAL_SD_CSV)
df_geom = pd.read_csv(GEOM_CSV)

df_cli["patient_id"] = df_cli["patient_id"].astype(str).str.strip()
df_geom["patient_id"] = df_geom["patient_id"].astype(str).str.strip()

df = df_cli.merge(df_geom, on="patient_id", how="left")

for c in ["Edad", "tamano_nodulo_mm", "profundidad_min_pleura_mm", "profundidad_centroidal_pleura_mm"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df.to_csv(OUT_CSV, index=False)
print(f"OK -> {OUT_CSV}")
print(df[["tamano_nodulo_mm","profundidad_min_pleura_mm","profundidad_centroidal_pleura_mm"]].isna().mean())