from pathlib import Path
import json
CANDIDATES=[Path('/mnt/homeGPU/mcribilles/tfm/external_data/LUNA16'),Path('/mnt/homeGPU/mcribilles/external_data/LUNA16'),Path('/mnt/homeGPU/mcribilles/external_data/LIDC-IDRI'),Path('/mnt/homeGPU/mcribilles/datasets/LUNA16'),Path('/mnt/homeGPU/mcribilles/datasets/LIDC-IDRI')]
found=[str(p) for p in CANDIDATES if p.exists()]
report={'strategy':'rescue_line3_luna_lidc_domain_pretraining','candidate_paths':[str(p) for p in CANDIDATES],'found_paths':found,'status':'ready_to_implement_training' if found else 'blocked_missing_external_dataset','note':'No se descarga automaticamente porque son datos externos; hay que preparar LUNA16/LIDC-IDRI localmente y documentar licencia/acceso.'}
Path('preflight_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
if not found:
    raise SystemExit(2)
