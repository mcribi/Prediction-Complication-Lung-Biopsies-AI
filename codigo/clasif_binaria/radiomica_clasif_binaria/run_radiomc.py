# -*- coding: utf-8 -*-
import os
import sys


import os
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# cualquier crash nativo deje traza
import faulthandler
faulthandler.enable()
print("[PY] PID={} HOST={} CWD={}".format(os.getpid(), os.uname().nodename, os.getcwd()), flush=True)

import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from datetime import datetime
import joblib
import random
import time
import unicodedata
import traceback
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold



# Utilidades de normalización/column matching
def _norm_text(s):
    s = ''.join(ch for ch in unicodedata.normalize('NFKC', str(s)) if ch != '\ufeff')
    return s.strip()

def _key(s):
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()

def _rename_cols_like(df, target_name, *aliases):
    want_keys = {_key(target_name)} | {_key(a) for a in aliases}
    mapping = {_key(c): c for c in df.columns}
    for k in want_keys:
        if k in mapping:
            real = mapping[k]
            if real != target_name:
                df.rename(columns={real: target_name}, inplace=True)
            return True
    return False

def _norm_id_series(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r'[, ]+', '', regex=True)
    s = s.str.replace(r'_(seg|lung|nodule)$', '', regex=True, flags=0)
    return s

# ----------------------------
# Rutas robustas basadas en este archivo
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG_FILE = os.path.join(SCRIPT_DIR, "valid_configurations_radiomics_unique.json")
OUTPUT_DB = os.path.join(PROJECT_ROOT, "radiomic_results", "sqlite_fixed", "results_radiomics_unique.db")
MODELS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "radiomic")

os.makedirs(os.path.dirname(OUTPUT_DB), exist_ok=True)
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

def resolve_path(p):
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

# ----------------------------
# Índice de configuración
# ----------------------------
if len(sys.argv) < 2:
    print("Usage: python radiomica/run_radiomc.py <CONFIG_INDEX>", flush=True)
    sys.exit(1)

config_index = int(sys.argv[1])
print("Running configuration index: {}".format(config_index), flush=True)
print("CONFIG_FILE: {}".format(CONFIG_FILE), flush=True)
print("OUTPUT_DB  : {}".format(OUTPUT_DB), flush=True)
print("MODELS_DIR : {}".format(MODELS_OUTPUT_DIR), flush=True)

# ----------------------------
# Cargar configuración
# ----------------------------
with open(CONFIG_FILE, "r") as f:
    all_configs = json.load(f)

config = all_configs[config_index]
print("Loaded configuration:\n{}".format(json.dumps(config, indent=2, ensure_ascii=False)), flush=True)

# unpack
features_csv = resolve_path(config.get("features_csv"))
labels_csv   = resolve_path(config.get("labels_csv"))
mapping_csv  = resolve_path(config.get("mapping_csv")) if config.get("mapping_csv") else None

id_col         = config.get("id_col", "patient_id")
label_col      = config.get("label_col", "label_complicacion")
label_positive = config.get("label_positive", "S")
use_masks      = config.get("use_masks", "both_merge")
n_splits       = int(config.get("n_splits", 5))
random_state   = int(config.get("random_state", 42))
shuffle_folds  = bool(config.get("shuffle_folds", True))

scaler_choice   = config.get("scaler")  # None | "StandardScaler" | "MinMaxScaler"
classifier_name = config.get("classifier")
classifier_params = {k: v for k, v in config.items()
                     if k not in ["features_csv","labels_csv","mapping_csv","id_col","label_col","label_positive",
                                  "use_masks","n_splits","random_state","shuffle_folds",
                                  "scaler","classifier",
                                  "dimred","preproc_names"]} 


print("[INFO] features_csv -> {}".format(features_csv), flush=True)
print("[INFO] labels_csv   -> {}".format(labels_csv), flush=True)
if mapping_csv:
    print("[INFO] mapping_csv  -> {}".format(mapping_csv), flush=True)

#reducion de dimensionalidad a probar 
dimred_choice = config.get("dimred", "none")  # "none" | "pca_99" | "pca_95" | "pca_90" | "varth_0.01"
preproc_names = config.get("preproc_names")

# ----------------------------
# DB Utils
# ----------------------------
def init_db():
    try:
        with sqlite3.connect(OUTPUT_DB) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_index INTEGER,
                    features_csv TEXT,
                    labels_csv TEXT,
                    use_masks TEXT,
                    classifier TEXT,
                    scaler TEXT,
                    classifier_params_json TEXT,
                    fold INTEGER,
                    n_train INTEGER,
                    n_test INTEGER,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    sensitivity REAL,
                    specificity REAL,
                    gmean REAL,
                    auc REAL,
                    tn INTEGER,
                    fp INTEGER,
                    fn INTEGER,
                    tp INTEGER,
                    timestamp TEXT,
                    preproc_names TEXT  -- NUEVO
                )
            ''')
            # Si la tabla ya existía sin la columna, la añadimos
            c.execute("PRAGMA table_info(experiments);")
            cols = [row[1] for row in c.fetchall()]
            if "preproc_names" not in cols:
                try:
                    c.execute("ALTER TABLE experiments ADD COLUMN preproc_names TEXT;")
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        print("Database initialized.", flush=True)
    except sqlite3.OperationalError as e:
        print("Error initializing database: {}".format(e), flush=True)
        print("Assuming database already exists. Skipping initialization.", flush=True)

def insert_result_db(fold_idx, n_train, n_test, metrics, cm_counts):
    # Valor a guardar: lista JSON de preprocesados usados en esta config
    preproc_names_json = json.dumps(preproc_names or [])

    while True:
        try:
            with sqlite3.connect(OUTPUT_DB) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO experiments (
                        config_index, features_csv, labels_csv, use_masks,
                        classifier, scaler, classifier_params_json, fold,
                        n_train, n_test,
                        accuracy, precision, recall, f1,
                        sensitivity, specificity, gmean, auc,
                        tn, fp, fn, tp,
                        timestamp, preproc_names
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config_index,
                    features_csv,
                    labels_csv,
                    use_masks,
                    classifier_name,
                    scaler_choice,
                    json.dumps(classifier_params),
                    fold_idx,
                    n_train, n_test,
                    metrics.get("accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("sensitivity"),
                    metrics.get("specificity"),
                    metrics.get("gmean"),
                    metrics.get("auc"),
                    cm_counts.get("tn"),
                    cm_counts.get("fp"),
                    cm_counts.get("fn"),
                    cm_counts.get("tp"),
                    datetime.now().isoformat(),
                    preproc_names_json
                ))
                conn.commit()
                return
        except sqlite3.OperationalError as e:
            random_wait = random.randint(5, 20)
            print("Database error: {}. Retrying in {} seconds...".format(e, random_wait), flush=True)
            time.sleep(random_wait)

init_db()

# ----------------------------
# Carga y preparación de datos
# ----------------------------
try:
    print("Loading features CSV…", flush=True)
    assert os.path.exists(features_csv), "No existe features_csv: {}".format(features_csv)
    df_feats = pd.read_csv(features_csv)

    if preproc_names:
        if "preproc_name" not in df_feats.columns:
            raise ValueError("Se pidió filtrar por preproc_names, pero el CSV no tiene columna 'preproc_name'.")
        df_feats = df_feats[df_feats["preproc_name"].isin(preproc_names)].copy()
        if df_feats.empty:
            raise ValueError(f"No hay filas tras filtrar por preproc_names={preproc_names}")

    print("Loading labels CSV…", flush=True)
    assert os.path.exists(labels_csv), "No existe labels_csv: {}".format(labels_csv)
    df_labels = pd.read_csv(labels_csv, sep=None, engine='python', encoding='utf-8-sig')
    df_labels.columns = [_norm_text(c) for c in df_labels.columns]

    ok_id  = _rename_cols_like(df_labels, id_col, "patient_id", "id paciente", "id_paciente", "paciente_id", "id")
    ok_lbl = _rename_cols_like(df_labels, label_col, "label_complicacion", "complicacion")
    if not (ok_id and ok_lbl):
        print("[ERROR] Columnas en labels:", list(df_labels.columns), flush=True)
        raise ValueError("labels_csv debe contener columnas '{}' y '{}' (se aceptan variantes)".format(id_col, label_col))

    if mapping_csv and os.path.exists(mapping_csv):
        mapdf = pd.read_csv(mapping_csv)
        assert "patient_id" in mapdf.columns and id_col in mapdf.columns, \
            "mapping_csv debe tener columnas 'patient_id' y '{}'".format(id_col)
        mapdf = mapdf.dropna(subset=[id_col])
        mapdf[id_col] = mapdf[id_col].astype(str).str.strip()
        df_feats = df_feats.merge(mapdf[["patient_id", id_col]], on="patient_id", how="left")
        missing_map = df_feats[id_col].isna().sum()
        if missing_map:
            print("[WARN] {} patient_id sin mapping; no tendrán etiqueta si no coinciden directamente.".format(missing_map), flush=True)
        join_left = id_col
    else:
        if id_col == "patient_id":
            join_left = "patient_id"
        else:
            if id_col in df_feats.columns:
                join_left = id_col
            else:
                join_left = "patient_id"
                print("[INFO] Uniendo con clinical usando '{}' (sin mapping).".format(join_left), flush=True)

    assert "patient_id" in df_feats.columns, "features_csv debe tener la columna 'patient_id'"

    print("[INFO] Merge directo patient_id <-> {} …".format(id_col), flush=True)
    tmp = df_feats.merge(df_labels[[id_col, label_col]], left_on="patient_id", right_on=id_col, how="left")
    n_lab_direct = tmp[label_col].notna().sum()
    print("[INFO] Etiquetas no nulas (directo): {}".format(n_lab_direct), flush=True)

    df_merged = tmp
    used_strategy = "direct"

    if n_lab_direct == 0:
        print("[INFO] Probo merge con IDs normalizados …", flush=True)
        df_feats["_id_norm"]  = _norm_id_series(df_feats["patient_id"])
        df_labels["_id_norm"] = _norm_id_series(df_labels[id_col])

        tmp2 = df_feats.merge(df_labels[["_id_norm", label_col]], on="_id_norm", how="left")
        n_lab_norm = tmp2[label_col].notna().sum()
        print("[INFO] Etiquetas no nulas (normalizado): {}".format(n_lab_norm), flush=True)

        if n_lab_norm > 0:
            df_merged = tmp2
            used_strategy = "normalized"
        else:
            if mapping_csv and os.path.exists(mapping_csv):
                print("[INFO] Aplicando mapping_csv: {}".format(mapping_csv), flush=True)
                mapdf = pd.read_csv(mapping_csv)
                assert "patient_id" in mapdf.columns and id_col in mapdf.columns, \
                    "mapping_csv debe tener columnas 'patient_id' y '{}'".format(id_col)
                mapdf[id_col] = mapdf[id_col].astype(str).str.strip()
                df_feats2 = df_feats.merge(mapdf[["patient_id", id_col]], on="patient_id", how="left")
                tmp3 = df_feats2.merge(df_labels[[id_col, label_col]], on=id_col, how="left")
                n_lab_map = tmp3[label_col].notna().sum()
                print("[INFO] Etiquetas no nulas (mapping): {}".format(n_lab_map), flush=True)
                if n_lab_map > 0:
                    df_merged = tmp3
                    used_strategy = "mapping"
                else:
                    raise ValueError("Tras directo/normalizado/mapping no hay etiquetas; revisa mapping o los IDs.")
            else:
                raise ValueError("Tras directo y normalizado no hay etiquetas; añade mapping_csv o revisa id_col.")

    for aux in ["_id_norm", id_col]:
        if aux in df_merged.columns and aux != "patient_id":
            if aux == id_col and used_strategy in ("direct", "normalized"):
                df_merged.drop(columns=[aux], inplace=True)
            elif aux == "_id_norm":
                df_merged.drop(columns=["_id_norm"], inplace=True, errors="ignore")

    df_feats = df_merged

    n_total = df_feats["patient_id"].nunique()
    n_lab   = df_feats[df_feats[label_col].notna()]["patient_id"].nunique()
    print("[INFO] Estrategia de merge usada: {} | Etiquetas por paciente: {}/{}".format(used_strategy, n_lab, n_total), flush=True)

    df_feats = df_feats[df_feats[label_col].notna()].copy()
    if df_feats.empty:
        raise ValueError("No hay casos etiquetados tras el merge.")

    if "mask_kind" in df_feats.columns:
        if use_masks == "lung":
            df_feats = df_feats[df_feats["mask_kind"] == "lung"].copy()
            df_feats.drop(columns=["mask_kind"], inplace=True)
        elif use_masks == "nodule":
            df_feats = df_feats[df_feats["mask_kind"] == "nodule"].copy()
            df_feats.drop(columns=["mask_kind"], inplace=True)
        elif use_masks == "both_merge":
            base_cols = ["patient_id", "mask_kind", label_col]
            feat_cols = [c for c in df_feats.columns if c not in base_cols]
            parts = []
            for mk in ["lung", "nodule"]:
                sub = df_feats[df_feats["mask_kind"] == mk][["patient_id"] + feat_cols + [label_col]].copy()
                sub.drop(columns=[label_col], inplace=True, errors="ignore")
                sub = sub.add_prefix("{}_".format(mk))
                sub.rename(columns={"{}_patient_id".format(mk): "patient_id"}, inplace=True)
                parts.append(sub)
            df_wide = parts[0]
            for p in parts[1:]:
                df_wide = df_wide.merge(p, on="patient_id", how="outer")
            lbl = df_feats[["patient_id", label_col]].drop_duplicates(subset=["patient_id"])
            df_feats = df_wide.merge(lbl, on="patient_id", how="left")
        else:
            raise ValueError("use_masks debe ser 'lung', 'nodule' o 'both_merge'")

    if "patient_id" not in df_feats.columns:
        raise ValueError("No se encontró 'patient_id' en el dataset final.")
    if label_col not in df_feats.columns:
        raise ValueError("No se encontró la columna de etiqueta '{}' tras el merge.".format(label_col))

    y_str = df_feats[label_col].astype(str)
    y = (y_str == str(label_positive)).astype(int).values

    drop_cols = ["patient_id", label_col]

    # elimina columnas no numéricas informativas: elimina patient_id
    for extra_col in ["preproc_name", "mask_kind"]:
        if extra_col in df_feats.columns:
            drop_cols.append(extra_col)

    X_cols = [c for c in df_feats.columns if c not in drop_cols]
    X = df_feats[X_cols].values


    print("Dataset listo: n={} muestras, d={} features. Positivos={}, Negativos={}".format(
        len(df_feats), X.shape[1], int((y==1).sum()), int((y==0).sum())
    ), flush=True)

except Exception as e:
    print("[ERROR] Exception during data prep:", repr(e), flush=True)
    traceback.print_exc()
    sys.exit(1)

# ----------------------------
# Pipeline y entrenamiento
# ----------------------------
steps = [("imputer", SimpleImputer(strategy="median"))]

# --- Dimensionalidad ---
# Decodificamos dimred_choice
pca_map = {"pca_99": 0.99, "pca_95": 0.95, "pca_90": 0.90}
use_pca = dimred_choice in pca_map
use_vt  = (dimred_choice == "varth_0.01")

# ¿Necesitamos escalar por PCA?
scaler_before_dimred = None
if use_pca:
    # Para PCA, estandarizamos SIEMPRE ANTES de PCA
    scaler_before_dimred = StandardScaler()
    steps.append(("scaler_pca", scaler_before_dimred))
    steps.append(("pca", PCA(n_components=pca_map[dimred_choice], svd_solver="full", random_state=random_state)))
elif use_vt:
    steps.append(("vt", VarianceThreshold(threshold=0.01)))
# si dimred_choice == "none": no hacemos nada

# --- Escalado para el clasificador ---
# SVM necesita escalado. Si ya hemos puesto StandardScaler para PCA, eso vale.
force_scaler = (classifier_name == "SVM")

# Si NO usamos PCA:
if not use_pca:
    if scaler_choice == "StandardScaler" or force_scaler:
        steps.append(("scaler", StandardScaler()))
    elif scaler_choice == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))
    elif scaler_choice is None or scaler_choice == "None":
        pass
    else:
        raise ValueError("Scaler no soportado: {}".format(scaler_choice))
else:
    # Usamos PCA: ya hemos puesto StandardScaler antes de PCA.
    # Ignoramos cualquier scaler extra para evitar doble escalado.
    if scaler_choice in ("MinMaxScaler",):
        print("[WARN] Se ignora MinMaxScaler porque PCA ya usa StandardScaler previo.", flush=True)


# Si el clasificador es SVM, forzamos StandardScaler aunque la config ponga null
# force_scaler = False

# if classifier_name == "SVM":
#     force_scaler = True  # SVM necesita features escaladas

# if scaler_choice == "StandardScaler" or force_scaler:
#     steps.append(("scaler", StandardScaler()))
# elif scaler_choice == "MinMaxScaler":
#     steps.append(("scaler", MinMaxScaler()))
# elif scaler_choice is None or scaler_choice == "None":
#     pass
# else:
#     raise ValueError("Scaler no soportado: {}".format(scaler_choice))

# --- Selección de clasificador ---
if classifier_name == "RandomForest":
    clf = RandomForestClassifier(random_state=random_state, **classifier_params)

elif classifier_name == "GradientBoosting":
    clf = GradientBoostingClassifier(random_state=random_state, **classifier_params)

elif classifier_name == "SVM":
    # Tomamos y quitamos parámetros relevantes para evitar conflictos
    kernel = classifier_params.pop("kernel", "rbf")
    C = classifier_params.pop("C", 1.0)
    gamma = classifier_params.pop("gamma", "scale")
    tol = classifier_params.pop("tol", 1e-3)
    max_iter = classifier_params.pop("max_iter", 5000)

    if kernel == "linear":
        # MUCHO más rápido que SVC(kernel='linear') y no soporta probas (usaremos decision_function)
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=C, tol=tol, max_iter=max_iter, **classifier_params)
    else:
        # SVC sin probability. AUC usará decision_function.
        from sklearn.svm import SVC
        clf = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=False,   # para que no se quede pillado
            tol=tol,
            max_iter=max_iter,
            cache_size=1000,     # algo de cache ayuda
            **classifier_params
        )

elif classifier_name == "LogisticRegression":
    clf = LogisticRegression(random_state=random_state, max_iter=1000, **classifier_params)

elif classifier_name == "DecisionTree":
    clf = DecisionTreeClassifier(random_state=random_state, **classifier_params)

elif classifier_name == "KNN":
    clf = KNeighborsClassifier(**classifier_params)

elif classifier_name == "XGBoost":
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError("Fallo importando xgboost: {}. Quita XGBoost de la config o reinstálalo en el entorno.".format(e))
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                            random_state=random_state, **classifier_params)

elif classifier_name == "LightGBM":
    try:
        import lightgbm as lgb
    except Exception as e:
        raise RuntimeError("Fallo importando lightgbm: {}. Quita LightGBM de la config o reinstálalo en el entorno.".format(e))
    clf = lgb.LGBMClassifier(random_state=random_state, **classifier_params)

else:
    raise ValueError("Unsupported classifier: {}".format(classifier_name))

steps.append(("classifier", clf))
pipeline = Pipeline(steps)


skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_folds, random_state=random_state)

for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    print("\nFold {}/{}".format(fold_idx, n_splits), flush=True)
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Probabilidades o scores para AUC
    try:
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
            y_score = pipeline.decision_function(X_test)
        else:
            y_score = None
    except Exception:
        y_score = None

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1v = f1_score(y_test, y_pred, average="binary", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        gmean = np.sqrt(sensitivity * specificity)
    else:
        tn = fp = fn = tp = 0
        sensitivity = specificity = gmean = np.nan

    auc = roc_auc_score(y_test, y_score) if y_score is not None else np.nan

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1v,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "gmean": gmean,
        "auc": auc
    }
    cm_counts = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    print("Metrics: {}".format(json.dumps(metrics, indent=2)), flush=True)
    print("CM counts: {} | n_train={} n_test={}".format(cm_counts, len(tr_idx), len(te_idx)), flush=True)

    # Guardar en DB
    insert_result_db(fold_idx, len(tr_idx), len(te_idx), metrics, cm_counts)

    # Guardar modelo
    model_filename = "model_cfg{}_fold{}_{}_{}.pkl".format(
        config_index, fold_idx, classifier_name, use_masks
    )
    model_path = os.path.join(MODELS_OUTPUT_DIR, model_filename)
    joblib.dump(pipeline, model_path)
    print("Model saved to {}".format(model_path), flush=True)

print("\nAll folds processed and results saved to DB.", flush=True)
