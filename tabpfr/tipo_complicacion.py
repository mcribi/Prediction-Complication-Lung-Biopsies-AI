import os
import re
import unicodedata
import numpy as np
import pandas as pd

from tabpfn import TabPFNClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/mnt/homeGPU/mcribilles/tabpfn_models")


CSV_PATH = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data.csv"
DEVICE = "cpu"
N_SPLITS = 5

BASE_LABELS = ["neumotorax", "hemorragia", "derrame pleural"]  # puedes añadir más si aparecen

def norm_text(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # quita acentos
    s = re.sub(r"\s+", " ", s)
    return s

def split_list_cell(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "" or s.upper() == "X":
        return []
    parts = [norm_text(p) for p in s.split(",")]
    return [p for p in parts if p and p != "x"]

def multi_hot_text_column(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    items_series = df[col].apply(split_list_cell)
    vocab = sorted({item for items in items_series for item in items})
    out = pd.DataFrame(index=df.index)
    for item in vocab:
        safe = re.sub(r"[^a-z0-9]+", "_", item).strip("_")
        out[f"{prefix}__{safe}"] = items_series.apply(lambda lst: int(item in lst))
    return out

# 1) Carga
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# 2) Construye Y multilabel
comp = df["Complicación"].astype(str).str.strip().str.lower()
tipo = df["Tipo de complicación"].astype(str)

y_rows = []
for c, t in zip(comp, tipo):
    if c in ["no", "0", "false"]:
        y_rows.append([])  # no complicación: ninguna etiqueta activa
    else:
        y_rows.append(split_list_cell(t))

# Normaliza nombres de etiquetas base
base_norm = [norm_text(x) for x in BASE_LABELS]

Y = np.zeros((len(df), len(base_norm)), dtype=int)
for i, labs in enumerate(y_rows):
    s = set(labs)
    for j, b in enumerate(base_norm):
        Y[i, j] = int(b in s)

# Info de distribución por etiqueta
print("Distribución por etiqueta (positivos):")
for j, b in enumerate(base_norm):
    print(f"  {b}: {int(Y[:, j].sum())}")

# 3) Features: quita lo que no quieres usar o es fuga
drop_cols = ["patient_id", "Tipo de cáncer", "Complicación", "Tipo de complicación"]
X_base = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# Sexo -> 0/1
if "Sexo" in X_base.columns:
    X_base["Sexo"] = (
        X_base["Sexo"].astype(str).str.strip().str.lower()
        .map({"mujer": 0, "hombre": 1})
    )

# Edad numérica
if "Edad" in X_base.columns:
    X_base["Edad"] = pd.to_numeric(X_base["Edad"], errors="coerce")

parts = [X_base[["Sexo", "Edad"]].copy()]

if "Factor de riesgo" in df.columns:
    parts.append(multi_hot_text_column(df, "Factor de riesgo", "risk"))
if "Patología pulmonar" in df.columns:
    parts.append(multi_hot_text_column(df, "Patología pulmonar", "lung"))

X = pd.concat(parts, axis=1).fillna(0)

print("Shape X:", X.shape)

# 4) CV: para multi-label no hay estratificación perfecta sin librerías extra.
# Usamos estratificación por "hay alguna complicación" para que cada fold tenga mezcla.
has_any = (Y.sum(axis=1) > 0).astype(int)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 5) Entrena un TabPFN por etiqueta
all_metrics = {}
for j, label in enumerate(base_norm):
    y_bin = Y[:, j]

    # Si hay muy pocos positivos, CV no es fiable
    n_pos = int(y_bin.sum())
    n_neg = int((y_bin == 0).sum())
    print(f"\nEtiqueta: {label}  (pos={n_pos}, neg={n_neg})")

    if n_pos < 5:
        print("  Aviso: muy pocos positivos. Mejor agrupar/descartar esta etiqueta para evaluación.")
        continue

    aucs = []
    f1s = []
    cm_total = None

    for tr, te in skf.split(X, has_any):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y_bin[tr], y_bin[te]

        model = TabPFNClassifier(device=DEVICE)
        model.fit(Xtr, ytr)

        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # Puede fallar AUC si en un fold no hay positivos o no hay negativos
        try:
            aucs.append(roc_auc_score(yte, proba))
        except Exception:
            pass

        f1s.append(f1_score(yte, pred, zero_division=0))
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        cm_total = cm if cm_total is None else (cm_total + cm)

    all_metrics[label] = {
        "AUC_mean": float(np.mean(aucs)) if len(aucs) else None,
        "F1_mean": float(np.mean(f1s)),
        "CM_total": cm_total,
    }

    print("  AUC media:", "NA" if all_metrics[label]["AUC_mean"] is None else round(all_metrics[label]["AUC_mean"], 3))
    print("  F1  media:", round(all_metrics[label]["F1_mean"], 3))
    print("  CM total:\n", cm_total)

print("\nResumen etiquetas evaluadas:")
for k, v in all_metrics.items():
    if v["AUC_mean"] is None:
        print(f"  {k}: AUC NA, F1 {v['F1_mean']:.3f}")
    else:
        print(f"  {k}: AUC {v['AUC_mean']:.3f}, F1 {v['F1_mean']:.3f}")
