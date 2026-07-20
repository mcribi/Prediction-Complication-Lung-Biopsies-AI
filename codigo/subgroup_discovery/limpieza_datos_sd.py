import pandas as pd
import numpy as np

# Paths (ajusta a tu caso)
INPUT_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data_limpios.csv"
OUTPUT_CSV = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data_limpios_SD.csv"

# Config
ID_COL = "patient_id"
DROP_COLS = ["Sin_factor_de_riesgo", "Sin_patología_pulmonar"]

def to01(series: pd.Series) -> pd.Series:
    """Convierte a 0/1 int con NaN -> 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def safe_max(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """max fila a fila de columnas existentes (0 si no hay ninguna)."""
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 0:
        return pd.Series(0, index=df.index, dtype=int)
    x = df[cols].apply(to01)
    return x.max(axis=1).astype(int)

def merge_duplicate_base(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    Unifica duplicados tipo base, base.1, base.2... en base = max(...)
    """
    dup_cols = [c for c in df.columns if c == base or c.startswith(base + ".")]
    if len(dup_cols) <= 1:
        return df
    df[base] = safe_max(df, dup_cols)
    df.drop(columns=[c for c in dup_cols if c != base], inplace=True)
    return df

# Carga
df = pd.read_csv(INPUT_CSV)
df = df.replace(r"^\s*$", np.nan, regex=True)

# 1) Hipertensión pulmonar: agrupar en una (maneja duplicados con .1)
if "Hipertensión_pulmonar" in df.columns or any(c.startswith("Hipertensión_pulmonar.") for c in df.columns):
    df = merge_duplicate_base(df, "Hipertensión_pulmonar")

# 2) Variables agregadas
df["Fibrosis_any"]   = safe_max(df, ["Fibrosis", "Fibrosis_pulmonar"])
df["Dislipemia_any"] = safe_max(df, ["Dislipemia", "Hipercolesterolemia", "Hiperlipemia"])
df["DM_any"]         = safe_max(df, ["DM", "DM1", "DM2"])

df["Tabac_any"]      = safe_max(df, ["Tabaquismo", "Extabaquismo", "Tabaquismo_pasivo"])
df["Alcohol_any"]    = safe_max(df, ["Alcoholismo", "Exalcoholismo"])

df["Enfisema_any"]   = safe_max(df, [
    "Enfisema",
    "Enfisema_centrolobulillar",
    "Enfisema_panacinar",
    "Enfisema_paraseptal",
])

# 3) Quitar columnas Sin_* que no quieres como descriptores
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")

# 4) CardioRisk (opcional) y Urologic
# Si NO lo quieres, comenta estas 2 líneas:
df["CardioRisk"] = safe_max(df, ["HTA", "Cardiopatía", "SCACEST", "Hiperuricemia"])
df["Urologic"]   = safe_max(df, ["HBP"])

# 5) (Opcional) eliminar columnas origen para reducir redundancia en SD
# Si prefieres mantenerlas, comenta este bloque.
cols_to_drop_after = [
    "Fibrosis", "Fibrosis_pulmonar",
    "Dislipemia", "Hipercolesterolemia", "Hiperlipemia",
    "DM", "DM1", "DM2",
    "Tabaquismo", "Extabaquismo", "Tabaquismo_pasivo",
    "Alcoholismo", "Exalcoholismo",
    "Enfisema", "Enfisema_centrolobulillar", "Enfisema_panacinar", "Enfisema_paraseptal",
    "HBP", "HTA", "Cardiopatía", "SCACEST", "Hiperuricemia",
]
df.drop(columns=[c for c in cols_to_drop_after if c in df.columns], inplace=True, errors="ignore")

# 6) Asegurar que las nuevas variables son 0/1 int
for c in ["Fibrosis_any", "Dislipemia_any", "DM_any", "Tabac_any", "Alcohol_any", "Enfisema_any", "CardioRisk", "Urologic", "Hipertensión_pulmonar"]:
    if c in df.columns:
        df[c] = to01(df[c])

# Guardar
df.to_csv(OUTPUT_CSV, index=False)
print(f"Guardado CSV listo para SD en: {OUTPUT_CSV}")
print("Columnas nuevas:", [c for c in ["Hipertensión_pulmonar","Fibrosis_any","Dislipemia_any","DM_any","Tabac_any","Alcohol_any","Enfisema_any","CardioRisk","Urologic"] if c in df.columns])