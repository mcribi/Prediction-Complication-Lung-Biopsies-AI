#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, itertools, argparse

def is_valid_dataset_dir(path: str, min_folds=5) -> bool:
    if not os.path.isdir(path):
        return False
    for i in range(min_folds):
        fdir = os.path.join(path, f"fold_{i}")
        if not os.path.isdir(fdir):
            return False
        if not os.path.isfile(os.path.join(fdir, "train.csv")):
            return False
        if not os.path.isfile(os.path.join(fdir, "test.csv")):
            return False
    return True

def discover_datasets(root: str, recursive: bool = True) -> list:
    """Encuentra carpetas con fold_0..fold_4/train.csv|test.csv."""
    found = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            if is_valid_dataset_dir(dirpath):
                found.append(os.path.abspath(dirpath))
    else:
        for d in sorted(os.listdir(root)):
            p = os.path.join(root, d)
            if is_valid_dataset_dir(p):
                found.append(os.path.abspath(p))
    return sorted(found)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_root", default="./datasets/", help="Raíz donde están los datasets con folds")
    ap.add_argument("--out_json", default="config/valid_configurations.auto.json")
    ap.add_argument("--recursive", action="store_true", help="Buscar recursivamente dentro de cv_root")
    ap.add_argument("--limit_per_model", type=int, default=0,
                    help="Si >0, limita el nº de combinaciones por (dataset, scaler, modelo)")
    args = ap.parse_args()

    # 1) Descubre datasets existentes con folds
    datasets = discover_datasets(args.cv_root, recursive=args.recursive)
    print(f"encontrados {len(datasets)} datasets válidos bajo {args.cv_root}")
    if not datasets:
        raise SystemExit("No se encontraron datasets con fold_0..fold_4.")

    # 2) Scalers
    scalers = [None, "StandardScaler", "MinMaxScaler"]

    # 3) Espacio de búsqueda ampliado (solo parámetros válidos para tu runner)
    search_space = {
        # ----------------- Lineales / márgenes -----------------
        "LogisticRegression": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],                  # (seguro con lbfgs/liblinear)
            "solver": ["lbfgs", "liblinear"],
            "class_weight": [None, "balanced"]
        },
        "SVM": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
            "class_weight": [None, "balanced"]
        },

        # ----------------- Árboles y ensamblados -----------------
        "DecisionTree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": [None, "sqrt", "log2"],
            "class_weight": [None, "balanced"]
        },
        "RandomForest": {
            "n_estimators": [100, 300, 600],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": [None, "sqrt", "log2"],
            "bootstrap": [True],
            "class_weight": [None, "balanced"]
        },
        "GradientBoosting": {
            "n_estimators": [100, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5],               # profundidad de los árboles base
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },

        # ----------------- KNN -----------------
        "KNN": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
            "p": [1, 2]                         # 1=Manhattan, 2=Euclídea
        },

        # ----------------- XGBoost -----------------
        # Tu runner usa: xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **params)
        "XGBoost": {
            "n_estimators": [200, 500, 800],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
            "min_child_weight": [1, 5, 10],
            "reg_lambda": [0.0, 1.0, 5.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "gamma": [0.0, 0.1, 1.0]
        },

        # ----------------- LightGBM -----------------
        # Tu runner usa: lgb.LGBMClassifier(random_state=42, **params)
        "LightGBM": {
            "n_estimators": [300, 800, 1200],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 10, 20],
            "min_child_samples": [10, 20, 50],
            "subsample": [0.7, 0.9, 1.0],       # alias: bagging_fraction
            "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0],
            "reg_alpha": [0.0, 0.5, 1.0]
        }
    }

    # 4) Generar configs
    all_cfgs = []
    for ds in datasets:
        for scaler in scalers:
            for model, grid in search_space.items():
                keys = list(grid.keys())
                vals = [grid[k] for k in keys]
                combos_iter = itertools.product(*vals) if keys else [()]

                # Recorte opcional por modelo para no explotar combinatoria
                if args.limit_per_model and args.limit_per_model > 0:
                    # Consumo parcial sin aleatoriedad (primeros N)
                    combos_iter = itertools.islice(itertools.product(*vals), args.limit_per_model)

                for combo in combos_iter:
                    params = dict(zip(keys, combo)) if keys else {}
                    cfg = {
                        **params,
                        "dataset_path": ds,
                        "classifier": model,
                        "scaler": scaler
                    }
                    all_cfgs.append(cfg)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_cfgs, f, indent=2, ensure_ascii=False)

    print(f" guardadas {len(all_cfgs)} configuraciones en {args.out_json}")

if __name__ == "__main__":
    main()
