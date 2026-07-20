import argparse
import os
import math
import numpy as np
import pandas as pd

from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# ---------------------------
# utilidades de métricas
# ---------------------------
def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = math.sqrt(max(tpr,0)*max(tnr,0))
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None and len(np.unique(y_true)) == 2 else np.nan
    return acc, f1, tpr, tnr, gmean, auc

# ---------------------------
# modelos y grids
# ---------------------------
def build_models_and_grids(random_state):
    models = []

    models.append((
        "LogisticRegression",
        LogisticRegression(max_iter=5000, random_state=random_state),
        {
            "clf__penalty": ["l2"],
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs", "liblinear"],
            "clf__class_weight": [None, "balanced"]
        },
        True  # necesita escalado
    ))

    models.append((
        "SVC_RBF",
        SVC(probability=True, random_state=random_state),
        {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto"],
            "clf__class_weight": [None, "balanced"]
        },
        True
    ))

    base_linsvc = LinearSVC(random_state=random_state, max_iter=5000)
    models.append((
        "CalibratedLinearSVC",
        CalibratedClassifierCV(estimator=base_linsvc, method="sigmoid", cv=3),
        {
            "clf__estimator__C": [0.01, 0.1, 1.0, 10.0],
            "clf__estimator__class_weight": [None, "balanced"]
        },
        True
    ))

    models.append((
        "RandomForest",
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        {
            "clf__n_estimators": [200, 500, 1000],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__class_weight": [None, "balanced"]
        },
        False
    ))

    models.append((
        "ExtraTrees",
        ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
        {
            "clf__n_estimators": [300, 600, 1000],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None],
            "clf__class_weight": [None, "balanced"]
        },
        False
    ))

    models.append((
        "GradientBoosting",
        GradientBoostingClassifier(random_state=random_state),
        {
            "clf__n_estimators": [100, 300, 600],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_depth": [2, 3, 5],
            "clf__subsample": [1.0, 0.7]
        },
        False
    ))

    models.append((
        "KNeighbors",
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": [3, 5, 7, 11, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        },
        True
    ))

    models.append((
        "GaussianNB",
        GaussianNB(),
        {
            # sin hiperparámetros principales
        },
        False
    ))

    return models

# ---------------------------
# preparación de datos
# ---------------------------
def prepare_features(df, target_col, drop_cols):
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")

    # fuerza numérico
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    num_cols = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
        ],
        remainder="drop"
    )
    return X, y, preprocessor, num_cols

# ---------------------------
# loop principal
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta al CSV tabular")
    parser.add_argument("--target", default="Complicacion_binaria", help="Columna objetivo (0/1)")
    parser.add_argument("--preprocessed_dir", default="", help="Se registra en la columna preprocessed_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out_csv", default="resultados_ml_grid.csv")
    args = parser.parse_args()

    # columnas a retirar de X (además del target)
    forbidden = [
        "Id_paciente",
        "Derrame_pleural_leve",
        "Hemorragia",
        "Hemorragia_leve",
        "Neumotórax",
        "Sin_complicación",
        "Complicacion_binaria",  # por si acaso aparece duplicada al hacer drop
    ]

    df = pd.read_csv(args.csv)

    # preparar X/y
    X, y, _, num_cols = prepare_features(df, args.target, forbidden)

    # modelos (sin reducción de dimensionalidad)
    models = build_models_and_grids(random_state=args.seed)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        ytr, yva = y[tr_idx], y[va_idx]

        for model_name, clf, grid, needs_scaler in models:
            # Preprocesado: imputación (+ escalado si aplica)
            steps_num = [("imputer", SimpleImputer(strategy="median"))]
            if needs_scaler:
                steps_num.append(("scaler", StandardScaler()))

            preproc = ColumnTransformer(
                transformers=[("num", Pipeline(steps=steps_num), num_cols)],
                remainder="drop"
            )

            pipe = Pipeline(steps=[
                ("pre", preproc),
                ("clf", clf)
            ])

            # producto cartesiano del grid
            if not grid:
                param_dicts = [dict()]
            else:
                keys = list(grid.keys())
                values = [grid[k] for k in keys]
                from itertools import product
                param_dicts = [dict(zip(keys, combo)) for combo in product(*values)]

            for params in param_dicts:
                pipe.set_params(**params)
                pipe.fit(Xtr, ytr)

                # probabilidades (o score calibrado)
                def get_proba(est, X_):
                    try:
                        return est.predict_proba(X_)[:, 1]
                    except Exception:
                        try:
                            s = est.decision_function(X_)
                            return 1.0 / (1.0 + np.exp(-s))
                        except Exception:
                            return None

                ytr_pred = pipe.predict(Xtr)
                ytr_proba = get_proba(pipe, Xtr)
                yva_pred = pipe.predict(Xva)
                yva_proba = get_proba(pipe, Xva)

                tr_acc, tr_f1, tr_tpr, tr_tnr, tr_gm, tr_auc = compute_metrics(ytr, ytr_pred, ytr_proba)
                va_acc, va_f1, va_tpr, va_tnr, va_gm, va_auc = compute_metrics(yva, yva_pred, yva_proba)

                flat_params = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in params.items()}
                common = {
                    "model": model_name,
                    "dimred": "none",
                    "fold": fold_idx,
                    "seed": args.seed,
                    "preprocessed_dir": args.preprocessed_dir,
                    "batch_size": np.nan,
                    "learning_rate": np.nan,
                    "weight_decay": np.nan,
                    "dropout_prob": np.nan,
                }
                common.update(flat_params)

                rows.append({**common, "set": "train", "accuracy": tr_acc, "f1": tr_f1, "tpr": tr_tpr, "tnr": tr_tnr, "gmean": tr_gm, "auc": tr_auc})
                rows.append({**common, "set": "val",   "accuracy": va_acc, "f1": va_f1, "tpr": va_tpr, "tnr": va_tnr, "gmean": va_gm, "auc": va_auc})

    df_out = pd.DataFrame(rows)

    first_cols = ["fold", "set", "accuracy", "f1", "tpr", "tnr", "gmean", "auc",
                  "batch_size", "learning_rate", "weight_decay", "dropout_prob",
                  "seed", "preprocessed_dir", "model", "dimred"]
    other_cols = [c for c in df_out.columns if c not in first_cols]
    df_out = df_out[first_cols + other_cols]

    df_out.to_csv(args.out_csv, index=False)
    print(f"[OK] Guardado: {args.out_csv}  | filas: {len(df_out)}")

if __name__ == "__main__":
    main()
