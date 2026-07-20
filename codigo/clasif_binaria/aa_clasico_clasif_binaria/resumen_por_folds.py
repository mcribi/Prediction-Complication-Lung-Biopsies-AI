import pandas as pd

# Carga tus resultados fold a fold
df = pd.read_csv("./aa_clasico/resultados_ml_grid.csv")

# --- Si quieres promediar SOLO por modelo y set (mezclando hiperparámetros):
summary = (
    df.groupby(["model", "set"])
      .agg({
          "accuracy": ["mean", "std"],
          "f1": ["mean", "std"],
          "tpr": ["mean", "std"],
          "tnr": ["mean", "std"],
          "gmean": ["mean", "std"],
          "auc": ["mean", "std"],
      })
      .reset_index()
)

# columnas más limpias
summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

# guarda
summary.to_csv("./aa_clasico/resumen_por_modelo.csv", index=False)
print("Guardado resumen_por_modelo.csv")
