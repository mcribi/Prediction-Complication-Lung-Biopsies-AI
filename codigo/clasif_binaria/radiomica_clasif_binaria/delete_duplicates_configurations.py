import json
from collections import Counter

IN_PATH  = "./radiomica/valid_configurations_radiomics_fixed.json"
OUT_PATH = "./radiomica/valid_configurations_radiomics_unique.json"

# Cargar
with open(IN_PATH, "r", encoding="utf-8") as f:
    configurations = json.load(f)

# Canonicalizador: ordena claves, mantiene orden de listas y respeta tipos (1 vs 1.0)
def canon(obj):
    # separators quita espacios para que la cadena sea 100% determinista
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)

seen = set()
unique_configurations = []
for cfg in configurations:
    s = canon(cfg)
    if s not in seen:
        seen.add(s)
        unique_configurations.append(cfg)

# Guardar (el contenido se guarda tal cual, sin reordenar listas)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(unique_configurations, f, indent=2, ensure_ascii=False, sort_keys=True)

print(f"Configuraciones originales: {len(configurations)}")
print(f"Configuraciones únicas: {len(unique_configurations)}")
print(f"Duplicados eliminados: {len(configurations) - len(unique_configurations)}")

# (Opcional) reporte de duplicados exactos por frecuencia
freq = Counter(canon(c) for c in configurations)
dupes = sum(1 for v in freq.values() if v > 1)
print(f"Entradas con al menos un duplicado: {dupes}")
