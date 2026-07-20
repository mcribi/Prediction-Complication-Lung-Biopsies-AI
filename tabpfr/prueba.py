import os
import re
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "/mnt/homeGPU/mcribilles/tabpfn_models")

CSV_PATH = "/mnt/homeGPU/mcribilles/tfm/clinical_data/clinical_data.csv"  # cambia si tu archivo se llama distinto
DEVICE = "cpu"  # o "cuda" si tienes GPU

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def split_list_cell(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "" or s.upper() == "X":
        return []
    # separa por coma
    parts = [normalize_text(p) for p in s.split(",")]
    return [p for p in parts if p and p != "x"]

def multi_hot(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    items_series = df[col].apply(split_list_cell)
    vocab = sorted({item for items in items_series for item in items})
    out = pd.DataFrame(index=df.index)
    for item in vocab:
        safe = re.sub(r"[^a-z0-9]+", "_", item).strip("_")
        out[f"{prefix}__{safe}"] = items_series.apply(lambda lst: int(item in lst))
    return out

# 1) carga
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

# 2) objetivo
y = (
    df["Complicación"].astype(str).str.strip().str.lower()
    .map({"sí": 1, "si": 1, "no": 0})
)
mask = y.notna()
df = df.loc[mask].copy()
y = y.loc[mask].astype(int)

print("Distribución y:", y.value_counts().to_dict())

# 3) features base
# quitamos lo que no quieres usar o es fuga
drop_cols = ["patient_id", "Tipo de cáncer", "Tipo de complicación", "Complicación"]
X_base = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# Sexo: Mujer/Hombre -> 0/1 (si hay otros valores, se dejan como NaN y luego se rellenan)
sexo = X_base.get("Sexo")
if sexo is not None:
    X_base["Sexo"] = (
        sexo.astype(str).str.strip().str.lower()
        .map({"mujer": 0, "hombre": 1})
    )

# Edad a numérico
if "Edad" in X_base.columns:
    X_base["Edad"] = pd.to_numeric(X_base["Edad"], errors="coerce")

# 4) multi-hot para columnas listas
X_parts = [X_base[["Sexo", "Edad"]].copy()]

if "Factor de riesgo" in X_base.columns:
    X_parts.append(multi_hot(df, "Factor de riesgo", "risk"))
if "Patología pulmonar" in X_base.columns:
    X_parts.append(multi_hot(df, "Patología pulmonar", "lung"))

X = pd.concat(X_parts, axis=1)

# missing -> 0 (sexo desconocido, edad desconocida, etc.)
X = X.fillna(0)

print("Shape X:", X.shape)

# 5) modelo + CV
model = TabPFNClassifier(device=DEVICE)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

aucs, baccs = [], []
cm_total = None

for tr, te in skf.split(X, y):
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]

    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    aucs.append(roc_auc_score(yte, proba))
    baccs.append(balanced_accuracy_score(yte, pred))

    cm = confusion_matrix(yte, pred)
    cm_total = cm if cm_total is None else (cm_total + cm)

print(f"AUC media (5-fold): {sum(aucs)/len(aucs):.3f}")
print(f"Balanced Acc media (5-fold): {sum(baccs)/len(baccs):.3f}")
print("Matriz de confusión total (suma folds):")
print(cm_total)
