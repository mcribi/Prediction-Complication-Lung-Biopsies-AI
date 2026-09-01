from __future__ import annotations

from pathlib import Path

import torch

import image_two_outputs.dl_nested_cv_two_outputs as image
from multimodal_two_outputs.multimodal_common import select_best_complete_candidate
import pretrained_two_outputs.pretrained_freeze_two_outputs as pretrained_freeze
import pretrained_two_outputs.pretrained_nested_cv_pilots as pretrained


def last_linear_out_features(model: torch.nn.Module) -> int:
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    if not layers:
        raise AssertionError(f"No linear layer found in {type(model).__name__}")
    return int(layers[-1].out_features)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the smoke job")
    expected_labels = ["Hemorragia", "Neumotórax"]
    if (
        image.TARGET_LABELS != expected_labels
        or pretrained.TARGET_LABELS != expected_labels
        or pretrained_freeze.TARGET_LABELS != expected_labels
    ):
        raise AssertionError("Target labels are not the exact two-output endpoint")

    clinical_source = image.pd.read_csv(image.base.CLINICAL_CSV, dtype={"patient_id": str})
    if len(clinical_source) != 210:
        raise AssertionError(f"Expected 210 clinical source rows, found {len(clinical_source)}")
    labels = image.base.build_multilabel_dataframe(image.base.CLINICAL_CSV)
    excluded_off_target = set(clinical_source["patient_id"]) - set(labels["patient_id"])
    if len(labels) != 209 or excluded_off_target != {"27HASD"}:
        raise AssertionError(
            f"Unexpected target-eligible cohort: size={len(labels)}, excluded={sorted(excluded_off_target)}"
        )

    preprocessings = image.base.discover_preprocessings()
    expected_preprocessing_names = {
        f"resize_{shape}{suffix}"
        for shape in ("cube64", "cube128", "small", "medium")
        for suffix in ("", "_hu_m300_1400", "_hu_m600_1500", "_multiwindowing_separadas")
    }
    observed_names = {row["name"] for row in preprocessings}
    if observed_names != expected_preprocessing_names:
        raise AssertionError(
            f"Unexpected preprocessing inventory: missing={sorted(expected_preprocessing_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_preprocessing_names)}"
        )

    cohort_sizes = {}
    for profile in ("focused", "densenet_light"):
        input_configs = image.resolve_input_profiles(profile)
        common_ids = image.base.collect_common_ids(labels, preprocessings, input_configs)
        cohort_sizes[profile] = len(common_ids)
        if len(common_ids) != 204:
            raise AssertionError(f"Expected 204 effective imaging patients for {profile}, found {len(common_ids)}")

    for model_name in image.MODEL_NAMES:
        model = image.build_model(model_name, 1, len(expected_labels))
        if last_linear_out_features(model) != 2:
            raise AssertionError(f"Image model {model_name} does not have two outputs")
        del model

    for source in ("medicalnet", "genesis_chest_ct"):
        report_dir = Path("/tmp") / f"dl2_{source}"
        model = pretrained.build_pretrained_model(source, 1, len(expected_labels), report_dir)
        if last_linear_out_features(model) != 2:
            raise AssertionError(f"Pretrained model {source} does not have two outputs")
        del model

        freeze_report_dir = Path("/tmp") / f"dl2_freeze_{source}"
        freeze_model = pretrained_freeze.build_pretrained_model(
            source, 1, len(expected_labels), freeze_report_dir
        )
        if last_linear_out_features(freeze_model) != 2:
            raise AssertionError(f"Freeze model {source} does not have two outputs")
        freeze_model.train()
        pretrained_freeze.set_frozen_encoder_batchnorm_eval(freeze_model, source)
        head_prefix = "fc" if source == "medicalnet" else "classifier"
        encoder_batch_norms = [
            module
            for name, module in freeze_model.named_modules()
            if not name.startswith(head_prefix)
            and isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        if not encoder_batch_norms or any(module.training for module in encoder_batch_norms):
            raise AssertionError(f"Frozen encoder BatchNorm is active for {source}")
        del freeze_model

    pretrained_freeze.LUNA_CHECKPOINT = pretrained_freeze.find_latest_luna_checkpoint()
    luna_model = pretrained_freeze.build_pretrained_model(
        "luna16_resnet34", 1, len(expected_labels), Path("/tmp/dl2_luna16")
    )
    if last_linear_out_features(luna_model) != 2:
        raise AssertionError("LUNA16 model does not have two outputs")
    del luna_model

    batch_norm = pretrained.ContBatchNorm3d(2)
    batch_norm.eval()
    mean_before = batch_norm.running_mean.clone()
    variance_before = batch_norm.running_var.clone()
    with torch.no_grad():
        batch_norm(torch.randn(3, 2, 4, 4, 4))
    if not torch.equal(batch_norm.running_mean, mean_before):
        raise AssertionError("Genesis BatchNorm changed running_mean during evaluation")
    if not torch.equal(batch_norm.running_var, variance_before):
        raise AssertionError("Genesis BatchNorm changed running_var during evaluation")

    freeze_batch_norm = pretrained_freeze.ContBatchNorm3d(2)
    freeze_batch_norm.eval()
    freeze_mean_before = freeze_batch_norm.running_mean.clone()
    with torch.no_grad():
        freeze_batch_norm(torch.randn(3, 2, 4, 4, 4))
    if not torch.equal(freeze_batch_norm.running_mean, freeze_mean_before):
        raise AssertionError("Freeze Genesis BatchNorm changed state during evaluation")

    selected = select_best_complete_candidate(
        [
            {
                "status": "ok",
                "num_folds_completed": 5,
                "all_folds_present": True,
                "config_id": "lower",
                "tuned_oof_two_outputs_f1_micro": 0.4,
                "tuned_oof_two_outputs_f1_macro": 0.5,
            },
            {
                "status": "ok",
                "num_folds_completed": 5,
                "all_folds_present": True,
                "config_id": "higher",
                "tuned_oof_two_outputs_f1_micro": 0.6,
                "tuned_oof_two_outputs_f1_macro": 0.5,
            },
        ]
    )
    if selected["config_id"] != "higher":
        raise AssertionError("Multimodal selector did not use tuned two-output OOF metrics")

    print(
        "PROJECT_SETUP_OK "
        f"clinical_source=210 target_eligible=209 excluded=27HASD preprocessings={len(preprocessings)} "
        f"cohorts={cohort_sizes} image_models={len(image.MODEL_NAMES)} pretrained_models=5",
        flush=True,
    )


if __name__ == "__main__":
    main()
