from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import CHANNELS, ExperimentConfig, SINGLE_HORIZON_VARIANTS
from .cohort import training_cases, validation_case
from .datasets import HorizonEEGDataset
from .training import load_checkpoint, predict_loader, train_model


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "pr_auc": float("nan"),
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    pr_auc = (
        float(average_precision_score(y_true, y_score))
        if np.any(y_true == 1)
        else float("nan")
    )
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": pr_auc,
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    prediction_probabilities: np.ndarray,
) -> dict[str, float]:
    """Report 3-state metrics and the distinct preictal-vs-interictal claim."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction_probabilities = np.asarray(prediction_probabilities, dtype=np.float64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    result: dict[str, float] = {
        "state_accuracy": float(np.mean(y_true == y_pred)),
        "state_macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
    }
    for index, label in enumerate(("interictal", "preictal", "ictal")):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        result[f"state_{label}_sensitivity"] = float(tp / (tp + fn)) if tp + fn else float("nan")
        result[f"state_{label}_specificity"] = float(tn / (tn + fp)) if tn + fp else float("nan")

    binary_mask = (y_true == 0) | (y_true == 1)
    binary_true = (y_true[binary_mask] == 1).astype(np.int64)
    if probabilities.shape != (len(y_true), 3):
        raise ValueError("Expected one 3-class probability vector per state label.")
    if prediction_probabilities.shape != (len(y_true), 2):
        raise ValueError("Expected one 2-class prediction vector per state label.")
    binary_pred = prediction_probabilities[binary_mask].argmax(axis=1)
    binary_scores = prediction_probabilities[binary_mask, 1]
    binary = _binary_metrics(binary_true, binary_pred, binary_scores)
    result.update({f"prediction_{key}": value for key, value in binary.items()})
    result["state_preictal_pr_auc"] = (
        float(average_precision_score((y_true == 1).astype(int), probabilities[:, 1]))
        if np.any(y_true == 1)
        else float("nan")
    )
    return result


def _patient_ids(cache_dir: Path) -> list[str]:
    return sorted(
        path.name
        for path in cache_dir.glob("chb*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )


def _validate_cache_configs(
    cache_dir: Path, patient_ids: list[str], config: ExperimentConfig
) -> None:
    expected = config.to_dict()
    preprocessing_fields = (
        "sampling_rate",
        "window_seconds",
        "overlap",
        "sph_minutes",
        "interictal_buffer_minutes",
        "horizons_minutes",
    )
    for patient in patient_ids:
        manifest_path = cache_dir / patient / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = manifest["config"]
        mismatched = [
            field for field in preprocessing_fields if actual.get(field) != expected[field]
        ]
        if mismatched:
            raise ValueError(
                f"Prepared cache for {patient} disagrees on {mismatched}; "
                "re-run prepare with one shared preprocessing configuration."
            )


def evaluate_loso(
    cache_dir: Path,
    output_dir: Path,
    config: ExperimentConfig,
    model_variants: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
    max_folds: int | None = None,
    include_baselines: bool = False,
    samples_per_epoch: int | None = None,
) -> pd.DataFrame:
    """Train and evaluate subject-held-out folds across horizons and ablations."""
    patient_ids = _patient_ids(cache_dir)
    if len(patient_ids) < 3:
        raise ValueError(
            f"LOSO with a separate patient-level validation set requires at least three "
            f"prepared subjects; found {len(patient_ids)}."
        )
    _validate_cache_configs(cache_dir, patient_ids, config)
    variants = model_variants or (
        "temporal_only",
        "temporal_spectral",
        "multi_domain",
        "subject_invariant",
    )
    if include_baselines:
        variants = (*variants, "cnn", "lstm", "cnn_lstm")
    variants = tuple(dict.fromkeys(variants))
    unknown = sorted(set(variants) - set(SINGLE_HORIZON_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown model variants: {unknown}")
    horizons = horizons or config.horizons_minutes
    if any(horizon not in config.horizons_minutes for horizon in horizons):
        raise ValueError(
            f"Requested horizons must have been prepared: {config.horizons_minutes}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when provided.")
    folds = patient_ids[:max_folds] if max_folds else patient_ids
    rows: list[dict[str, object]] = []
    total_runs = len(folds) * len(horizons) * len(variants)
    run_number = 0

    for horizon in horizons:
        for test_index, test_subject in enumerate(folds):
            validation_subject = validation_case(patient_ids, test_subject)
            train_subjects = training_cases(patient_ids, test_subject, validation_subject)
            if len(train_subjects) < 2 and "subject_invariant" in variants:
                raise ValueError(
                    "Subject-invariant training needs two training patients after "
                    "reserving validation and test subjects."
                )
            subject_indices = {subject: index for index, subject in enumerate(train_subjects)}
            train_data = [
                HorizonEEGDataset(
                    cache_dir / subject,
                    horizon,
                    subject_index=subject_indices[subject],
                )
                for subject in train_subjects
            ]
            validation_data = HorizonEEGDataset(
                cache_dir / validation_subject, horizon, subject_index=len(train_subjects)
            )
            test_data = HorizonEEGDataset(
                cache_dir / test_subject, horizon, subject_index=len(train_subjects) + 1
            )
            for variant in variants:
                run_number += 1
                fold_dir = output_dir / f"h{horizon}" / variant / test_subject
                checkpoint_path = fold_dir / "model.pt"
                print(
                    f"[{run_number}/{total_runs}] SOP={horizon}m "
                    f"model={variant} test={test_subject} validation={validation_subject}"
                )
                fold_config = ExperimentConfig.from_dict(
                    {
                        **config.to_dict(),
                        "preictal_minutes": horizon,
                        "model_variant": variant,
                    }
                )
                train_result = train_model(
                    train_data,
                    validation_data,
                    fold_config,
                    variant,
                    checkpoint_path,
                    samples_per_epoch=samples_per_epoch,
                )
                model, _, device = load_checkpoint(checkpoint_path)
                test_loader = DataLoader(
                    test_data, batch_size=config.batch_size, shuffle=False, num_workers=0
                )
                (
                    y_true,
                    y_pred,
                    probabilities,
                    prediction_probabilities,
                ) = predict_loader(model, test_loader, device)
                metrics = calculate_metrics(
                    y_true, y_pred, probabilities, prediction_probabilities
                )
                row: dict[str, object] = {
                    "horizon_minutes": horizon,
                    "model_variant": variant,
                    "test_subject": test_subject,
                    "validation_subject": validation_subject,
                    "train_subjects": ",".join(train_subjects),
                    "best_epoch": train_result["best_epoch"],
                    "validation_prediction_f1": train_result[
                        "best_validation_prediction_f1"
                    ],
                    "validation_selection_f1": train_result[
                        "best_validation_selection_f1"
                    ],
                    "checkpoint": str(checkpoint_path),
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_dir / "loso_per_subject.csv", index=False)

    results = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    metric_columns = [
        column
        for column in results.select_dtypes(include=[np.number]).columns
        if column not in {"horizon_minutes", "best_epoch"}
    ]
    for (horizon, variant), group in results.groupby(["horizon_minutes", "model_variant"]):
        row: dict[str, object] = {
            "horizon_minutes": int(horizon),
            "model_variant": str(variant),
            "subjects": int(group["test_subject"].nunique()),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else math.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "loso_summary.csv", index=False)
    results.to_csv(output_dir / "loso_per_subject.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "models": variants,
                "horizons_minutes": list(horizons),
                "test_subjects": folds,
                "channels": list(CHANNELS),
                "protocol": "LOSO with a distinct patient-level validation subject",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results
