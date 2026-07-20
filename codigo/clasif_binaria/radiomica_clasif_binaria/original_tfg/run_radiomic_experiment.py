#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import joblib
import random
import time

# ===============================
# CONFIG
# ===============================
CONFIG_FILE = os.environ.get("CONFIG_FILE", "./config/valid_configurations.HUGE.json")
#CONFIG_FILE = "./config/valid_configurations.HUGE.json"
OUTPUT_DB = "./radiomic_results/sqlite_fixed/results_ml2.db"
MODELS_OUTPUT_DIR = "./models/fixed/radiomic_ml2"

os.makedirs("./radiomic_results", exist_ok=True)
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

# ===============================
# Get SLURM Array Index
# ===============================
if len(sys.argv) < 2:
    print("Usage: python run_experiment.py <CONFIG_INDEX>")
    sys.exit(1)

config_index = int(sys.argv[1])
print(f"✅ Running configuration index: {config_index}")

# ===============================
# Load Configuration
# ===============================
with open(CONFIG_FILE, "r") as f:
    all_configs = json.load(f)

config = all_configs[config_index]
print(f"✅ Loaded configuration:\n{json.dumps(config, indent=2)}")

# Split config into dataset and classifier params
dataset_path = config.get("dataset_path")
classifier_name = config.get("classifier")
scaler_choice = config.get("scaler")
classifier_params = {k: v for k, v in config.items() if k not in ["classifier", "scaler", "dataset_path"]}
classifier_params_json = json.dumps(classifier_params)


# ===============================
# DB Utils
# ===============================
def init_db():
    try:
        with sqlite3.connect(OUTPUT_DB) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_index INTEGER,
                    dataset_path TEXT,
                    classifier TEXT,
                    scaler TEXT,
                    classifier_params_json TEXT,
                    fold TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1 REAL,
                    gmean REAL,
                    timestamp TEXT
                )
            ''')
            conn.commit()
        print("✅ Database initialized.")
    except sqlite3.OperationalError as e:
        print(f"❌ Error initializing database: {e}")
        print(f"Assuming database already exists. Skipping initialization.")




def insert_result_db(config_index, dataset_path, classifier_name, scaler_choice, classifier_params_json, fold_name, metrics):
    while True:
        try:
            with sqlite3.connect(OUTPUT_DB) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO experiments (
                        config_index, dataset_path, classifier, scaler, classifier_params_json, fold,
                        accuracy, precision, recall, f1, gmean, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config_index,
                    dataset_path,
                    classifier_name,
                    scaler_choice,
                    classifier_params_json,
                    fold_name,
                    metrics.get("accuracy"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("gmean"),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return
        except sqlite3.OperationalError as e:
            random_wait = random.randint(5, 20)
            print(f"❌ Database error: {e}. Retrying in {random_wait} seconds...")
            time.sleep(random_wait)


# ===============================
# Initialize DB (once per job)
# ===============================
init_db()

if not dataset_path or not os.path.isdir(dataset_path):
    print(f"⚠️ Dataset path no encontrado: {dataset_path}. Salto esta config.")
    sys.exit(0)

if classifier_name == "LightGBM":
    # En LightGBM, 'colsample_bytree' -> 'feature_fraction'; 'subsample' -> 'bagging_fraction'
    if "colsample_bytree" in classifier_params and "feature_fraction" not in classifier_params:
        classifier_params["feature_fraction"] = classifier_params.pop("colsample_bytree")
    if "subsample" in classifier_params and "bagging_fraction" not in classifier_params:
        classifier_params["bagging_fraction"] = classifier_params.pop("subsample")


# ===============================
# Detect Folds
# ===============================
fold_dirs = sorted([
    os.path.join(dataset_path, d)
    for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])
print(f"✅ Folds found: {fold_dirs}")

if len(fold_dirs) < 5:
    print("❌ Not enough folds found. At least 5 are required.")
    sys.exit(1)

# ===============================
# Loop over Folds
# ===============================
for fold_dir in fold_dirs:
    fold_name = os.path.basename(fold_dir)
    print(f"\n🔎 Processing {fold_name}")

    # ----------------------------
    # Load Data
    # ----------------------------
    train_file = os.path.join(fold_dir, "train.csv")
    test_file = os.path.join(fold_dir, "test.csv")

    df_train = pd.read_csv(train_file, index_col=0)
    df_test = pd.read_csv(test_file, index_col=0)

    target_col = "label_complicacion"
    feature_cols = [col for col in df_train.columns if not col.startswith("label_")]

    X_train = df_train[feature_cols].values
    y_train = (df_train[target_col].values == 'S').astype(int)
    X_test = df_test[feature_cols].values
    y_test = (df_test[target_col].values == 'S').astype(int)


    # ----------------------------
    # Build Pipeline
    # ----------------------------
    steps = []

    if scaler_choice == "StandardScaler":
        steps.append(("scaler", StandardScaler()))
    elif scaler_choice == "MinMaxScaler":
        steps.append(("scaler", MinMaxScaler()))

    if classifier_name == "RandomForest":
        clf = RandomForestClassifier(random_state=42, **classifier_params)
    elif classifier_name == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=42, **classifier_params)
    # elif classifier_name == "SVM":
    #     clf = SVC(probability=True, random_state=42, **classifier_params)
    elif classifier_name == "SVM":
        # Defaults rápidos
        svm_defaults = dict(
            probability=False,      # MUY importante: acelera muchísimo
            random_state=42,
            max_iter=5000,          # límite duro
            tol=1e-3,               # tolerancia un poco más laxa
            cache_size=1000         # cache grande en MB acelera kernel RBF
        )
        # Mezcla con lo que venga del JSON (si el JSON define prob/max_iter, los respeta)
        svm_defaults.update(classifier_params)

        # Heurística anti-lentitud: si RBF y el problema es grande, forzar 'linear'
        # (define “grande” como >5000 features o >2000 ejemplos; ajusta si quieres)
        n_train, n_feat = X_train.shape
        k = svm_defaults.get("kernel", "rbf")
        if k == "rbf" and (n_feat > 5000 or n_train > 2000):
            print(f"⚠️ SVM kernel=rbf con problema grande ({n_train}x{n_feat}). Forzando kernel='linear'.")
            svm_defaults["kernel"] = "linear"

        clf = SVC(**svm_defaults)

    elif classifier_name == "LogisticRegression":
        clf = LogisticRegression(random_state=42, max_iter=500, **classifier_params)
    elif classifier_name == "DecisionTree":
        clf = DecisionTreeClassifier(random_state=42, **classifier_params)
    elif classifier_name == "KNN":
        clf = KNeighborsClassifier(**classifier_params)
    elif classifier_name == "XGBoost":
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **classifier_params)
    elif classifier_name == "LightGBM":
        clf = lgb.LGBMClassifier(random_state=42, **classifier_params)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    steps.append(("classifier", clf))
    pipeline = Pipeline(steps)

    # ----------------------------
    # Train
    # ----------------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # ----------------------------
    # Metrics
    # ----------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        gmean = np.sqrt(sensitivity * specificity)
    else:
        gmean = None

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "gmean": gmean
    }

    print(f"✅ Metrics: {json.dumps(metrics, indent=2)}")

    # ----------------------------
    # Save to DB
    # ----------------------------
    classifier_params_json = json.dumps(classifier_params)
    insert_result_db(
        config_index,
        dataset_path,
        classifier_name,
        scaler_choice,
        classifier_params_json,
        fold_name,
        metrics
    )


    # ----------------------------
    # Save trained model
    # ----------------------------
    model_filename = f"model_config{config_index}_{fold_name}.pkl"
    model_path = os.path.join(MODELS_OUTPUT_DIR, model_filename)
    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to {model_path}")

print("\n✅ All folds processed and results saved to DB.")
