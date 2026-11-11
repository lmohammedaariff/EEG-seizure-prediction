from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader

from .config import CHANNELS, MULTIHORIZON_MODEL_VARIANTS, ExperimentConfig
from .cohort import training_cases, validation_case
from .datasets import MultiHorizonEEGDataset
from .event_metrics import calibrate_thresholds, event_forecast_metrics
from .explanation_audit import audit_event_explanations
from .training import (
    load_checkpoint,
    predict_multihorizon_loader,
    train_multihorizon_model,
)


MULTIHORIZON_VARIANTS = MULTIHORIZON_MODEL_VARIANTS


def _prediction_frames(
    dataset: MultiHorizonEEGDataset,
    predictions: dict[str, np.ndarray],
    horizon_index: int,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    record_indices = predictions["recording_indices"]
    labels = predictions["horizon_labels"][:, horizon_index]
    scores = predictions["horizon_probabilities"][:, horizon_index]
    starts = predictions["starts_seconds"]
    for index, recording in enumerate(dataset.recording_names):
        selected = record_indices == index
        frames[recording.lower()] = pd.DataFrame(
            {
                "time_seconds": starts[selected],
                "label": labels[selected],
                "score": scores[selected],
            }
        )
    return frames


def _absolute_events(
    dataset: MultiHorizonEEGDataset,
) -> list[tuple[float, float]]:
    events: list[tuple[float, float]] = []
    for recording, intervals in dataset.seizures_by_recording.items():
        offset = dataset.recording_offsets_seconds.get(recording)
        if offset is None:
            raise ValueError(f"No time offset found for annotated recording {recording}.")
        events.extend((offset + onset, offset + end) for onset, end in intervals)
    return sorted(events)


def _window_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    valid = labels >= 0
    y_true = labels[valid]
    y_score = scores[valid]
    y_pred = (y_score >= threshold).astype(np.int64)
    positives = y_true == 1
    negatives = y_true == 0
    tp = int(np.sum(positives & (y_pred == 1)))
    tn = int(np.sum(negatives & (y_pred == 0)))
    fn = int(np.sum(positives & (y_pred == 0)))
    fp = int(np.sum(negatives & (y_pred == 1)))
    return {
        "window_sensitivity": tp / (tp + fn) if tp + fn else math.nan,
        "window_specificity": tn / (tn + fp) if tn + fp else math.nan,
        "window_f1": (
            float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else math.nan
        ),
        "window_pr_auc": (
            float(average_precision_score(y_true, y_score))
            if np.any(positives)
            else math.nan
        ),
    }


def _state_metrics(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    labels = predictions["state_labels"]
    predicted = predictions["state_predictions"]
    valid = labels >= 0
    if not np.any(valid):
        return {"state_accuracy": math.nan, "state_macro_f1": math.nan}
    return {
        "state_accuracy": float(np.mean(labels[valid] == predicted[valid])),
        "state_macro_f1": float(
            f1_score(labels[valid], predicted[valid], labels=[0, 1, 2],
                     average="macro", zero_division=0)
        ),
    }


def evaluate_multihorizon_loso(
    cache_dir: Path,
    output_dir: Path,
    config: ExperimentConfig,
    model_variants: tuple[str, ...] = MULTIHORIZON_VARIANTS,
    max_folds: int | None = None,
    samples_per_epoch: int | None = 6000,
    threshold_candidates: int = 81,
    resume: bool = False,
) -> pd.DataFrame:
    """Train shared multi-horizon heads and evaluate alarms on held-out patients."""
    variants = tuple(dict.fromkeys(model_variants))
    unknown = sorted(set(variants) - set(MULTIHORIZON_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown multi-horizon model variants: {unknown}")
    if len(config.horizons_minutes) < 2:
        raise ValueError("Multi-horizon evaluation needs at least two prepared horizons.")
    patient_ids = sorted(
        path.name
        for path in cache_dir.glob("chb*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if len(patient_ids) < 4:
        raise ValueError(
            "Patient-held-out evaluation with a separate validation patient requires "
            "at least four prepared patients."
        )
    _validate_cache_configs(cache_dir, patient_ids, config)
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be positive when provided.")
    test_subjects = patient_ids[:max_folds] if max_folds else patient_ids
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_label_audit(cache_dir, output_dir, patient_ids, config.horizons_minutes)
    results_path = output_dir / "multihorizon_loso_per_patient.csv"
    explanation_path = output_dir / "explanation_audit_per_event.csv"
    rows: list[dict[str, Any]] = []
    explanation_rows: list[dict[str, Any]] = []
    completed_pairs: set[tuple[str, str]] = set()
    if resume and results_path.is_file():
        existing = pd.read_csv(results_path)
        if not existing.empty:
            expected_subjects = set(test_subjects)
            if not set(existing["model_variant"]).issubset(variants):
                raise ValueError("Existing results contain models not selected for resume.")
            if not set(existing["test_subject"]).issubset(expected_subjects):
                raise ValueError("Existing results contain test patients outside this run.")
            if (
                set(existing["horizon_minutes"].astype(int))
                - set(config.horizons_minutes)
                or set(existing["sph_minutes"].astype(int)) != {config.sph_minutes}
                or set(existing["false_alarm_budget_per_hour"].astype(float))
                != set(config.far_budgets_per_hour)
            ):
                raise ValueError("Existing results use incompatible horizons or alarm settings.")
            expected_metrics = {
                (int(horizon), float(budget))
                for horizon in config.horizons_minutes
                for budget in config.far_budgets_per_hour
            }
            for (variant, subject), group in existing.groupby(
                ["model_variant", "test_subject"]
            ):
                actual_metrics = {
                    (int(row.horizon_minutes), float(row.false_alarm_budget_per_hour))
                    for row in group.itertuples()
                }
                checkpoint = (
                    output_dir / "checkpoints" / variant / subject / "model.pt"
                )
                if (
                    actual_metrics == expected_metrics
                    and len(group) == len(expected_metrics)
                    and checkpoint.is_file()
                ):
                    completed_pairs.add((variant, subject))
                    rows.extend(group.to_dict(orient="records"))
        if explanation_path.is_file() and explanation_path.read_text(
            encoding="utf-8"
        ).strip():
            explanation_rows.extend(
                pd.read_csv(explanation_path).to_dict(orient="records")
            )
    audited_patients = {
        str(row["patient_id"])
        for row in explanation_rows
        if row.get("patient_id") is not None and not pd.isna(row.get("patient_id"))
    }
    total_runs = len(test_subjects) * len(variants)
    run_number = 0

    for fold_index, test_subject in enumerate(test_subjects):
        validation_subject = validation_case(patient_ids, test_subject)
        train_subjects = training_cases(patient_ids, test_subject, validation_subject)
        if len(train_subjects) < 2:
            raise ValueError(
                "Subject-invariant training needs at least two training patients "
                "after reserving validation and test subjects."
            )
        subject_indices = {subject: index for index, subject in enumerate(train_subjects)}
        train_datasets = [
            MultiHorizonEEGDataset(
                cache_dir / subject,
                config.horizons_minutes,
                subject_index=subject_indices[subject],
            )
            for subject in train_subjects
        ]
        validation_dataset = MultiHorizonEEGDataset(
            cache_dir / validation_subject,
            config.horizons_minutes,
            subject_index=len(train_subjects),
        )
        test_dataset = MultiHorizonEEGDataset(
            cache_dir / test_subject,
            config.horizons_minutes,
            subject_index=len(train_subjects) + 1,
        )
        _validate_dataset_horizons(
            [*train_datasets, validation_dataset, test_dataset],
            config.horizons_minutes,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        validation_events = _absolute_events(validation_dataset)
        test_events = _absolute_events(test_dataset)
        for model_variant in variants:
            run_number += 1
            fold_config = ExperimentConfig.from_dict(
                {
                    **config.to_dict(),
                    "model_variant": model_variant,
                    "preictal_minutes": max(config.horizons_minutes),
                }
            )
            checkpoint = (
                output_dir
                / "checkpoints"
                / model_variant
                / test_subject
                / "model.pt"
            )
            print(
                f"[{run_number}/{total_runs}] multi-horizon model={model_variant} "
                f"test={test_subject} validation={validation_subject} "
                f"train={len(train_subjects)} patients"
            )
            is_complete = (model_variant, test_subject) in completed_pairs
            if is_complete and (
                model_variant != "multi_horizon" or test_subject in audited_patients
            ):
                continue
            if is_complete:
                model, checkpoint_payload, device = load_checkpoint(
                    checkpoint, device=torch.device("cpu")
                )
                _validate_resumed_checkpoint(
                    checkpoint_payload,
                    fold_config,
                    model_variant,
                    train_subjects,
                    validation_subject,
                    samples_per_epoch,
                )
                test_predictions = predict_multihorizon_loader(
                    model, test_loader, device
                )
                for horizon_index in range(len(config.horizons_minutes)):
                    explanation_rows.extend(
                        {
                            **row,
                            "model_variant": model_variant,
                        }
                        for row in audit_event_explanations(
                            model,
                            test_dataset,
                            horizon_index,
                            fold_config,
                        )
                    )
                audited_patients.add(test_subject)
                del model
                continue
            if resume and checkpoint.is_file():
                # A valid checkpoint may outlive a partial CSV write. Reuse it
                # to rebuild missing metrics without fitting the fold again.
                model, checkpoint_payload, device = load_checkpoint(checkpoint)
                _validate_resumed_checkpoint(
                    checkpoint_payload,
                    fold_config,
                    model_variant,
                    train_subjects,
                    validation_subject,
                    samples_per_epoch,
                )
                training_result = {
                    "best_epoch": checkpoint_payload["best_epoch"],
                    "best_validation_mean_horizon_pr_auc": checkpoint_payload[
                        "best_validation_mean_horizon_pr_auc"
                    ],
                }
            else:
                training_result = train_multihorizon_model(
                    train_datasets,
                    validation_dataset,
                    fold_config,
                    model_variant,
                    checkpoint,
                    samples_per_epoch=samples_per_epoch,
                )
                model, _, device = load_checkpoint(checkpoint)
            validation_predictions = predict_multihorizon_loader(
                model, validation_loader, device
            )
            test_predictions = predict_multihorizon_loader(model, test_loader, device)
            state_metrics = _state_metrics(test_predictions)
            if model_variant == "multi_horizon" and test_subject not in audited_patients:
                for explanation_horizon_index in range(
                    len(config.horizons_minutes)
                ):
                    explanation_rows.extend(
                        {
                            **row,
                            "model_variant": model_variant,
                        }
                        for row in audit_event_explanations(
                            model,
                            test_dataset,
                            explanation_horizon_index,
                            fold_config,
                        )
                    )
                audited_patients.add(test_subject)

            for horizon_index, horizon in enumerate(config.horizons_minutes):
                validation_frames = _prediction_frames(
                    validation_dataset, validation_predictions, horizon_index
                )
                test_frames = _prediction_frames(
                    test_dataset, test_predictions, horizon_index
                )
                thresholds_by_budget = calibrate_thresholds(
                    validation_frames,
                    validation_dataset.recording_offsets_seconds,
                    validation_events,
                    horizon_minutes=horizon,
                    sph_minutes=config.sph_minutes,
                    false_alarm_budgets_per_hour=config.far_budgets_per_hour,
                    window_seconds=config.window_seconds,
                    alarm_merge_gap_minutes=config.alarm_merge_gap_minutes,
                    candidate_count=threshold_candidates,
                )
                labels = test_predictions["horizon_labels"][:, horizon_index]
                scores = test_predictions["horizon_probabilities"][:, horizon_index]
                for far_budget, (threshold, _) in thresholds_by_budget.items():
                    event_metrics = event_forecast_metrics(
                        test_frames,
                        test_dataset.recording_offsets_seconds,
                        test_events,
                        horizon_minutes=horizon,
                        sph_minutes=config.sph_minutes,
                        threshold=threshold,
                        window_seconds=config.window_seconds,
                        alarm_merge_gap_minutes=config.alarm_merge_gap_minutes,
                    )
                    row = {
                        "model_variant": model_variant,
                        "test_subject": test_subject,
                        "validation_subject": validation_subject,
                        "train_subjects": ",".join(train_subjects),
                        "horizon_minutes": int(horizon),
                        "sph_minutes": config.sph_minutes,
                        "false_alarm_budget_per_hour": far_budget,
                        "validation_threshold": threshold,
                        "best_epoch": training_result["best_epoch"],
                        "validation_mean_horizon_pr_auc": training_result[
                            "best_validation_mean_horizon_pr_auc"
                        ],
                        **state_metrics,
                        **_window_metrics(labels, scores, threshold),
                        **event_metrics,
                    }
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(results_path, index=False)
            del model

    results = pd.DataFrame(rows)
    metric_columns = (
        "event_sensitivity",
        "false_alarms_per_hour",
        "mean_lead_time_minutes",
        "median_lead_time_minutes",
        "window_sensitivity",
        "window_specificity",
        "window_f1",
        "window_pr_auc",
        "state_accuracy",
        "state_macro_f1",
    )
    summary_rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(
        ["model_variant", "horizon_minutes", "false_alarm_budget_per_hour"]
    ):
        summary: dict[str, Any] = {
            "model_variant": keys[0],
            "horizon_minutes": int(keys[1]),
            "false_alarm_budget_per_hour": float(keys[2]),
            "subjects": int(group["test_subject"].nunique()),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            summary[f"{metric}_mean"] = (
                float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
            )
            summary[f"{metric}_std"] = (
                float(np.nanstd(values, ddof=1))
                if np.isfinite(values).sum() > 1
                else math.nan
            )
            summary[f"{metric}_n"] = int(np.isfinite(values).sum())
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "multihorizon_loso_summary.csv", index=False)
    explanation_frame = pd.DataFrame(explanation_rows)
    explanation_frame.to_csv(
        output_dir / "explanation_audit_per_event.csv", index=False
    )
    if not explanation_frame.empty:
        explanation_metrics = (
            "channel_rank_spearman_mean",
            "top_channel_jaccard_mean",
            "top_channel_mask_risk_drop_mean",
            "random_channel_mask_risk_drop_mean",
            "top_channel_mask_advantage",
        )
        explanation_summary_rows = []
        for horizon, group in explanation_frame.groupby("horizon_minutes"):
            summary_row: dict[str, Any] = {
                "horizon_minutes": int(horizon),
                "patients": int(group["patient_id"].nunique()),
                "events": int(group["event_index_in_recording"].count()),
            }
            for metric in explanation_metrics:
                values = group[metric].to_numpy(dtype=float)
                summary_row[f"{metric}_mean"] = (
                    float(np.nanmean(values))
                    if np.isfinite(values).any()
                    else math.nan
                )
                summary_row[f"{metric}_std"] = (
                    float(np.nanstd(values, ddof=1))
                    if np.isfinite(values).sum() > 1
                    else math.nan
                )
            explanation_summary_rows.append(summary_row)
        pd.DataFrame(explanation_summary_rows).to_csv(
            output_dir / "explanation_audit_summary.csv", index=False
        )
    patient_audits: dict[str, dict[str, int]] = {}
    for patient_id in patient_ids:
        patient_manifest = json.loads(
            (cache_dir / patient_id / "manifest.json").read_text(encoding="utf-8")
        )
        patient_audits[patient_id] = {
            "edf_recordings": len(patient_manifest["records"]),
            "annotated_seizure_events": sum(
                len(events)
                for events in patient_manifest.get("seizures_by_recording", {}).values()
            ),
        }
    (output_dir / "multihorizon_manifest.json").write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "models": list(variants),
                "horizons_minutes": list(config.horizons_minutes),
                "test_subjects": test_subjects,
                "dataset_audit": {
                    "patients": patient_audits,
                    "total_edf_recordings": sum(
                        audit["edf_recordings"] for audit in patient_audits.values()
                    ),
                    "total_annotated_seizure_events": sum(
                        audit["annotated_seizure_events"]
                        for audit in patient_audits.values()
                    ),
                    "recording_time_source": (
                        "EDF header timestamps normalized to the earliest recording "
                        "when all are present; otherwise CHB-MIT summary clocks."
                    ),
                },
                "protocol": (
                    "Patient-held-out test folds; separate patient-level validation; "
                    "training uses only remaining patients."
                ),
                "alarm_protocol": {
                    "score_time": "end of each EEG window",
                    "forecast_interval": (
                        "[seizure onset - SOP - SPH, seizure onset - SPH)"
                    ),
                    "threshold_selection": (
                        "Validation patient only; maximize event sensitivity subject "
                        "to each false-alarm/hour budget."
                    ),
                    "alarm_merge_gap_minutes": config.alarm_merge_gap_minutes,
                    "false_alarm_denominator": (
                        "Union duration of horizon-labeled interictal windows; "
                        "ictal and ambiguous windows excluded."
                    ),
                    "threshold_grid_candidates": threshold_candidates,
                },
                "warning": (
                    "Retrospective CHB-MIT subset evaluation; not a clinically "
                    "validated alarm."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def _validate_dataset_horizons(
    datasets: list[MultiHorizonEEGDataset],
    expected: tuple[int, ...],
) -> None:
    for dataset in datasets:
        if tuple(dataset.horizons_minutes) != tuple(expected):
            raise ValueError(
                f"Dataset horizons for {dataset.patient_id} do not match {expected}."
            )


def _validate_resumed_checkpoint(
    payload: dict[str, object],
    config: ExperimentConfig,
    model_variant: str,
    train_subjects: list[str],
    validation_subject: str,
    samples_per_epoch: int | None,
) -> None:
    expected = {
        "config": config.to_dict(),
        "model_variant": model_variant,
        "train_subjects": train_subjects,
        "validation_subject": validation_subject,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Cannot resume {model_variant}: checkpoint {key} does not match "
                "the requested LOSO fold."
            )
    saved_samples = payload.get("training_samples_per_epoch")
    if saved_samples is not None and saved_samples != samples_per_epoch:
        raise ValueError(
            f"Cannot resume {model_variant}: training sample count changed from "
            f"{saved_samples} to {samples_per_epoch}."
        )


def _validate_cache_configs(
    cache_dir: Path,
    patient_ids: list[str],
    config: ExperimentConfig,
) -> None:
    expected = config.to_dict()
    required = (
        "sampling_rate",
        "window_seconds",
        "overlap",
        "sph_minutes",
        "interictal_buffer_minutes",
        "min_ictal_overlap_seconds",
        "horizons_minutes",
    )
    for patient_id in patient_ids:
        manifest = json.loads(
            (cache_dir / patient_id / "manifest.json").read_text(encoding="utf-8")
        )
        actual = manifest["config"]
        mismatched = [
            key
            for key in required
            if actual.get(key) != expected[key]
        ]
        if mismatched:
            raise ValueError(
                f"Prepared cache for {patient_id} disagrees on {mismatched}; "
                "relabel the cache with one shared configuration."
            )


def _write_label_audit(
    cache_dir: Path,
    output_dir: Path,
    patient_ids: list[str],
    horizons_minutes: tuple[int, ...],
) -> None:
    rows = []
    for patient_id in patient_ids:
        manifest = json.loads(
            (cache_dir / patient_id / "manifest.json").read_text(encoding="utf-8")
        )
        n_events = sum(
            len(events)
            for events in manifest.get("seizures_by_recording", {}).values()
        )
        for horizon in horizons_minutes:
            counts = np.zeros(3, dtype=np.int64)
            windows = 0
            for record in manifest["records"]:
                windows += int(record["windows"])
                values = record["class_counts_by_horizon"][str(horizon)]
                counts += np.asarray(values, dtype=np.int64)
            rows.append(
                {
                    "patient_id": patient_id,
                    "horizon_minutes": horizon,
                    "annotated_seizure_events": n_events,
                    "total_windows": windows,
                    "excluded_or_ambiguous_windows": int(windows - counts.sum()),
                    "interictal_windows": int(counts[0]),
                    "preictal_windows": int(counts[1]),
                    "ictal_windows": int(counts[2]),
                    "preictal_to_interictal_ratio": (
                        float(counts[1] / counts[0]) if counts[0] else math.nan
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "label_audit.csv", index=False)
