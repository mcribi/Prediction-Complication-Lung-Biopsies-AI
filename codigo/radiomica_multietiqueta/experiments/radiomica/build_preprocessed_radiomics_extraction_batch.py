import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build config list for preprocessed radiomics extraction batch.")
    parser.add_argument("--preproc_root", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    preproc_root = Path(args.preproc_root)
    out_json = Path(args.out_json)
    configs = []
    for preproc_dir in sorted([p for p in preproc_root.iterdir() if p.is_dir()]):
        nifti_images = preproc_dir / "nifti" / "images"
        if not nifti_images.exists():
            continue
        configs.append(
            {
                "preproc_name": preproc_dir.name,
                "preproc_dir": str(preproc_dir),
            }
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    print(f"configs_written: {len(configs)}")
    print(f"out_json: {out_json}")


if __name__ == "__main__":
    main()
