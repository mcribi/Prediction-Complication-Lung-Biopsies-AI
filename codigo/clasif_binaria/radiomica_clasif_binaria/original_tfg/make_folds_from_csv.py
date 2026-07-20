import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import unicodedata

def strip_accents_lower(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s_norm = unicodedata.normalize("NFKD", s)
    s_noacc = "".join(ch for ch in s_norm if not unicodedata.combining(ch))
    return s_noacc.strip().lower()

def detect_comp_column(cols):
    # intenta localizar "Complicación" ignorando acentos y mayúsculas
    for c in cols:
        if strip_accents_lower(c) == "complicacion":
            return c
    raise ValueError("No encuentro la columna 'Complicación' en el CSV de labels.")

def map_complicacion(val) -> str:
    v = strip_accents_lower(str(val))
    if v in {"si","sí","s","y","yes","true","1"}:
        return "S"
    if v in {"no","n","false","0"}:
        return "N"
    if v in {"", "nan", "none"}:
        return "N"
    # última defensa: si contiene 'si' (p.ej. texto libre)
    return "S" if "si" in v or "sí" in v else "N"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True,
                    help="CSV de features radiomicos (tiene 'patient_id' + columnas numéricas)")
    ap.add_argument("--labels_csv", required=True,
                    help="CSV de labels 'grande' con columnas en español (incluye 'patient_id' y 'Complicación')")
    ap.add_argument("--out_dir", required=True,
                    help="Directorio de salida con fold_0..fold_4")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1) Cargar features ----
    dfX = pd.read_csv(args.features_csv)
    if "patient_id" not in dfX.columns:
        raise ValueError("El CSV de features no tiene columna 'patient_id'")
    # limpia diagnostics_ si aparecieran
    diag_cols = [c for c in dfX.columns if c.startswith("diagnostics_")]
    dfX = dfX.drop(columns=diag_cols, errors="ignore")
    # asegurar tipo/espacios
    dfX["patient_id"] = dfX["patient_id"].astype(str).str.strip()

    # ---- 2) Cargar labels 'grande' y mapear Complicación -> S/N ----
    dfl = pd.read_csv(args.labels_csv)
    if "patient_id" not in dfl.columns:
        raise ValueError("El CSV de labels no tiene columna 'patient_id'")
    comp_col = detect_comp_column(dfl.columns)

    dfl["patient_id"] = dfl["patient_id"].astype(str).str.strip()
    dfl["label_complicacion"] = dfl[comp_col].apply(map_complicacion)

    # Resolver duplicados: si alguna fila del paciente tiene 'S' => 'S'; si no, 'N'
    dfl_min = (
        dfl[["patient_id", "label_complicacion"]]
        .groupby("patient_id", as_index=False)["label_complicacion"]
        .apply(lambda s: "S" if (s == "S").any() else "N")
        .rename(columns={"label_complicacion": "label_complicacion"})
    )

    # ---- 3) Merge features + labels ----
    df = pd.merge(dfX, dfl_min, on="patient_id", how="inner")
    if df.empty:
        raise ValueError("Tras el merge features+labels no quedan filas. ¿Coinciden los patient_id?")

    # Mantener solo numéricas/bool + patient_id + label
    keep = ["patient_id", "label_complicacion"]
    numeric_cols = df.select_dtypes(include=["number","bool"]).columns.tolist()
    df = df[keep + numeric_cols].copy()

    # Quitar columnas con NaN/Inf (si prefieres imputar, cámbialo por imputación)
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(axis=1, how="any")

    # ---- 4) Folds estratificados por label_complicacion ----
    y = (df["label_complicacion"] == "S").astype(int).values
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for i, (tr, te) in enumerate(skf.split(df, y)):
        fold_dir = os.path.join(args.out_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)
        df.iloc[tr].to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        df.iloc[te].to_csv(os.path.join(fold_dir, "test.csv"), index=False)
        print(f"fold_{i}: train={len(tr)}  test={len(te)}")

    print(f"OK -> Folds generados en {args.out_dir}")

if __name__ == "__main__":
    main()
