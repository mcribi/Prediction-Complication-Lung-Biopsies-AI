#!/usr/bin/env python3
"""Build the clinical feature table used by the multimodal models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path(
    "/mnt/homeGPU/mcribilles/tfm/clinical_data/210pacientes"
)
DEFAULT_INPUT = DEFAULT_DATA_DIR / "clinical_data_limpios_SD.csv"
DEFAULT_REFERENCE = DEFAULT_DATA_DIR / "clinical_data_limpios.csv"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "clinical_data_multimodal.csv"

IDENTIFIER = "patient_id"
OUTCOME_COLUMNS = {
    "Complicación",
    "Tipo de complicación",
    "Derrame_pleural",
    "Hemorragia",
    "Neumotórax",
    "Sin_complicación",
    "Complicacion_binaria",
    "Tipo de cáncer",
}

RISK_COLUMNS = [
    "AOS",
    "Aneurisma_esplénico",
    "Asma",
    "Craniofaringioma",
    "Cáncer_de_mama",
    "EPOC",
    "Encefalopatía",
    "Enfermedad_de_von_Willebrand",
    "Epilepsia",
    "FRVC",
    "Macroglobulinemia_de_Waldestrom",
    "NAMC",
    "Obesidad",
    "VIH",
    "Dislipemia_any",
    "DM_any",
    "Tabac_any",
    "Alcohol_any",
    "CardioRisk",
    "Urologic",
]

PULMONARY_COLUMNS = [
    "Asbestosis",
    "Atelectasia_crónica",
    "Bronquiectasias",
    "Enfisema_any",
    "Fibrosis_any",
    "Hipertensión_pulmonar",
    "Neumoconiosis",
    "Neumonía",
]

FEATURE_COLUMNS = [
    "Edad",
    "Sexo_binaria",
    "Tabac_any",
    "CardioRisk",
    "Enfisema_any",
    "Dislipemia_any",
    "Hipertensión_pulmonar",
    "DM_any",
    "Obesidad",
    "Fibrosis_any",
    "AOS",
    "Sin_factor_de_riesgo",
    "Sin_patología_pulmonar",
]

BINARY_FEATURES = [column for column in FEATURE_COLUMNS if column != "Edad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the clinical feature table for multimodal training."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def validate_identifier(df: pd.DataFrame, source: Path) -> None:
    if df[IDENTIFIER].isna().any():
        raise ValueError(f"Missing patient identifiers in {source}")
    if df[IDENTIFIER].duplicated().any():
        duplicated = df.loc[df[IDENTIFIER].duplicated(), IDENTIFIER].tolist()
        raise ValueError(f"Duplicated patient identifiers in {source}: {duplicated}")


def build_feature_table(source: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    require_columns(
        source,
        [IDENTIFIER, *RISK_COLUMNS, *PULMONARY_COLUMNS, *FEATURE_COLUMNS[:-2]],
        source_path,
    )

    prepared = source.copy()
    prepared["Sin_factor_de_riesgo"] = (
        prepared[RISK_COLUMNS].sum(axis=1) == 0
    ).astype("int8")
    prepared["Sin_patología_pulmonar"] = (
        prepared[PULMONARY_COLUMNS].sum(axis=1) == 0
    ).astype("int8")

    result = prepared.set_index(IDENTIFIER)[FEATURE_COLUMNS].copy()
    result.index.name = IDENTIFIER
    result["Edad"] = pd.to_numeric(result["Edad"], errors="raise")
    result[BINARY_FEATURES] = result[BINARY_FEATURES].astype("int8")
    return result


def validate_feature_table(
    result: pd.DataFrame,
    source: pd.DataFrame,
    reference: pd.DataFrame,
    reference_path: Path,
) -> None:
    if result.index.has_duplicates:
        raise ValueError("The output contains duplicated patient identifiers")
    if result.isna().any().any():
        raise ValueError("The output contains missing values")
    if result.index.tolist() != source[IDENTIFIER].tolist():
        raise ValueError("Patient order changed while preparing the output")

    forbidden = sorted(OUTCOME_COLUMNS.intersection(result.columns))
    if forbidden:
        raise ValueError(f"Outcome columns found in feature table: {forbidden}")
    if IDENTIFIER in result.columns:
        raise ValueError("patient_id must be an index, not a model feature")

    for column in BINARY_FEATURES:
        invalid = sorted(set(result[column].unique()) - {0, 1})
        if invalid:
            raise ValueError(f"Non-binary values in {column}: {invalid}")

    require_columns(
        reference,
        [IDENTIFIER, "Sin_factor_de_riesgo", "Sin_patología_pulmonar"],
        reference_path,
    )
    expected = reference.set_index(IDENTIFIER)[
        ["Sin_factor_de_riesgo", "Sin_patología_pulmonar"]
    ].loc[result.index]
    observed = result[["Sin_factor_de_riesgo", "Sin_patología_pulmonar"]]
    if not observed.astype("int64").equals(expected.astype("int64")):
        raise ValueError("Reconstructed absence indicators do not match reference data")


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.input)
    reference = pd.read_csv(args.reference)

    validate_identifier(source, args.input)
    validate_identifier(reference, args.reference)
    result = build_feature_table(source, args.input)
    validate_feature_table(result, source, reference, args.reference)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=True)

    print(f"Output: {args.output}")
    print(f"Patients: {len(result)}")
    print(f"Model features: {len(result.columns)}")
    print("Age remains unscaled and must be transformed inside each training fold")


if __name__ == "__main__":
    main()
