# summarize_results.py



#PONER NUMERO DE FOLDS PARA VER SI SE HAN HECHO LOS 5



import sqlite3, pandas as pd, numpy as np, sys

DB = "/mnt/homeGPU/mcribilles/tfm/codigo/radiomic_results/sqlite_fixed/results_radiomics.db"

with sqlite3.connect(DB) as conn:
    df = pd.read_sql_query("SELECT * FROM experiments", conn)

if df.empty:
    print("No hay filas en experiments.")
    sys.exit(0)

group_cols = ["config_index","features_csv","labels_csv","use_masks","classifier","scaler","classifier_params_json"]
metric_cols = ["accuracy","precision","recall","f1","sensitivity","specificity","gmean","auc"]

agg = df.groupby(group_cols)[metric_cols].agg(['mean','std']).reset_index()
# aplanar columnas (tuplas) a "mean_accuracy", etc.
agg.columns = ['_'.join([c for c in col if c]).strip('_') for col in agg.columns]
agg.rename(columns={"config_index_": "config_index"}, inplace=True)

with sqlite3.connect(DB) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_summaries AS
        SELECT 0 as id, 0 as config_index, '' as features_csv, '' as labels_csv, '' as use_masks,
               '' as classifier, '' as scaler, '' as classifier_params_json, 
               0 as n_folds,
               0.0 as mean_accuracy, 0.0 as std_accuracy,
               0.0 as mean_precision, 0.0 as std_precision,
               0.0 as mean_recall, 0.0 as std_recall,
               0.0 as mean_f1, 0.0 as std_f1,
               0.0 as mean_sensitivity, 0.0 as std_sensitivity,
               0.0 as mean_specificity, 0.0 as std_specificity,
               0.0 as mean_gmean, 0.0 as std_gmean,
               0.0 as mean_auc, 0.0 as std_auc,
               '' as timestamp
        """)
    # mejor limpiar e insertar de nuevo
    conn.execute("DELETE FROM experiment_summaries")
    for _, row in agg.iterrows():
        n_folds = int(df[df['config_index']==row['config_index']].shape[0])
        conn.execute("""
            INSERT INTO experiment_summaries (
                config_index, features_csv, labels_csv, use_masks,
                classifier, scaler, classifier_params_json, n_folds,
                mean_accuracy,  std_accuracy,
                mean_precision, std_precision,
                mean_recall,    std_recall,
                mean_f1,        std_f1,
                mean_sensitivity, std_sensitivity,
                mean_specificity, std_specificity,
                mean_gmean,       std_gmean,
                mean_auc,         std_auc,
                timestamp
            ) VALUES (?,?,?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?, datetime('now'))
        """, (
            int(row['config_index']),
            row['features_csv_'], row['labels_csv_'], row['use_masks_'],
            row['classifier_'], row['scaler_'], row['classifier_params_json_'],
            n_folds,
            float(row['accuracy_mean']),  float(row['accuracy_std']),
            float(row['precision_mean']), float(row['precision_std']),
            float(row['recall_mean']),    float(row['recall_std']),
            float(row['f1_mean']),        float(row['f1_std']),
            float(row['sensitivity_mean']), float(row['sensitivity_std']),
            float(row['specificity_mean']), float(row['specificity_std']),
            float(row['gmean_mean']),       float(row['gmean_std']),
            float(row['auc_mean']),         float(row['auc_std']),
        ))
    conn.commit()

print("Resúmenes escritos en experiment_summaries ")
