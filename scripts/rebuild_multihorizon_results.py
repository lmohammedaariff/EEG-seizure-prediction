"""Rebuild LOSO metrics from saved checkpoints without retraining any model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from seizure_prediction.config import MULTIHORIZON_MODEL_VARIANTS
from seizure_prediction.datasets import MultiHorizonEEGDataset
from seizure_prediction.event_metrics import calibrate_thresholds, event_forecast_metrics
from seizure_prediction.multihorizon_eval import (
    _absolute_events,
    _prediction_frames,
    _state_metrics,
    _validate_cache_configs,
    _validate_dataset_horizons,
    _validate_resumed_checkpoint,
    _window_metrics,
)
from seizure_prediction.cli import _read_cache_config
from seizure_prediction.cohort import training_cases, validation_case
from seizure_prediction.training import load_checkpoint, predict_multihorizon_loader


METRICS = (
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


def _aggregate(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(
        ["model_variant", "horizon_minutes", "false_alarm_budget_per_hour"]
    ):
        row: dict[str, Any] = {
            "model_variant": keys[0],
            "horizon_minutes": int(keys[1]),
            "false_alarm_budget_per_hour": float(keys[2]),
            "subjects": int(group["test_subject"].nunique()),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(values)
            row[f"{metric}_mean"] = float(np.nanmean(values)) if valid.any() else np.nan
            row[f"{metric}_std"] = (
                float(np.nanstd(values, ddof=1)) if valid.sum() > 1 else np.nan
            )
            row[f"{metric}_n"] = int(valid.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild(
    cache_dir: Path,
    output_dir: Path,
    samples_per_epoch: int = 6000,
    alarm_persistence_windows: int = 1,
    output_suffix: str = "",
    variants: list[str] | None = None,
) -> None:
    if alarm_persistence_windows < 1:
        raise ValueError("--alarm-persistence-windows must be at least one.")
    if output_suffix and not output_suffix.startswith("_"):
        output_suffix = f"_{output_suffix}"
    variants = variants or list(MULTIHORIZON_MODEL_VARIANTS)
    invalid = sorted(set(variants) - set(MULTIHORIZON_MODEL_VARIANTS))
    if invalid:
        raise ValueError(f"Unknown multi-horizon model variants: {invalid}.")
    config = replace(_read_cache_config(cache_dir), epochs=6, batch_size=64)
    cases = sorted(
        path.name
        for path in cache_dir.glob("chb*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if len(cases) < 2:
        raise ValueError(f"At least two prepared CHB-MIT cases are required, found {len(cases)}.")
    _validate_cache_configs(cache_dir, cases, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    resumable_results_path = output_dir / f"multihorizon_loso_per_patient{output_suffix}.csv"
    completed_folds: set[tuple[str, str]] = set()
    if resumable_results_path.is_file():
        previous = pd.read_csv(resumable_results_path)
        expected_protocol = previous.get("alarm_persistence_windows", pd.Series(dtype=int)).dropna().unique()
        if len(expected_protocol) and expected_protocol.tolist() != [alarm_persistence_windows]:
            raise ValueError(
                f"Existing results in {resumable_results_path} use a different persistence protocol; "
                "choose another --output-suffix."
            )
        rows.extend(previous.to_dict("records"))
        completed_folds = set(
            zip(previous["model_variant"].astype(str), previous["test_subject"].astype(str))
        )
    total = len(cases) * len(variants)
    run = 0

    for test_case in cases:
        validation = validation_case(cases, test_case)
        train = training_cases(cases, test_case, validation)
        train_index = {case: index for index, case in enumerate(train)}
        train_datasets = [
            MultiHorizonEEGDataset(
                cache_dir / case,
                config.horizons_minutes,
                subject_index=train_index[case],
            )
            for case in train
        ]
        validation_dataset = MultiHorizonEEGDataset(
            cache_dir / validation,
            config.horizons_minutes,
            subject_index=len(train),
        )
        test_dataset = MultiHorizonEEGDataset(
            cache_dir / test_case,
            config.horizons_minutes,
            subject_index=len(train) + 1,
        )
        _validate_dataset_horizons(
            [*train_datasets, validation_dataset, test_dataset], config.horizons_minutes
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        validation_events = _absolute_events(validation_dataset)
        test_events = _absolute_events(test_dataset)

        for variant in variants:
            run += 1
            if (variant, test_case) in completed_folds:
                print(f"Reusing completed fold [{run}/{total}]: {variant}/{test_case}", flush=True)
                continue
            checkpoint = output_dir / "checkpoints" / variant / test_case / "model.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
            fold_config = replace(
                config,
                model_variant=variant,
                preictal_minutes=max(config.horizons_minutes),
            )
            model, payload, device = load_checkpoint(checkpoint)
            _validate_resumed_checkpoint(
                payload, fold_config, variant, train, validation, samples_per_epoch
            )
            validation_predictions = predict_multihorizon_loader(
                model, validation_loader, device
            )
            test_predictions = predict_multihorizon_loader(model, test_loader, device)
            state_metrics = _state_metrics(test_predictions)

            for horizon_index, horizon in enumerate(config.horizons_minutes):
                validation_frames = _prediction_frames(
                    validation_dataset, validation_predictions, horizon_index
                )
                test_frames = _prediction_frames(test_dataset, test_predictions, horizon_index)
                thresholds = calibrate_thresholds(
                    validation_frames,
                    validation_dataset.recording_offsets_seconds,
                    validation_events,
                    horizon_minutes=horizon,
                    sph_minutes=config.sph_minutes,
                    false_alarm_budgets_per_hour=config.far_budgets_per_hour,
                    window_seconds=config.window_seconds,
                    alarm_merge_gap_minutes=config.alarm_merge_gap_minutes,
                    candidate_count=81,
                    alarm_persistence_windows=alarm_persistence_windows,
                )
                labels = test_predictions["horizon_labels"][:, horizon_index]
                scores = test_predictions["horizon_probabilities"][:, horizon_index]
                for budget, (threshold, _) in thresholds.items():
                    event_metrics = event_forecast_metrics(
                        test_frames,
                        test_dataset.recording_offsets_seconds,
                        test_events,
                        horizon_minutes=horizon,
                        sph_minutes=config.sph_minutes,
                        threshold=threshold,
                        window_seconds=config.window_seconds,
                        alarm_merge_gap_minutes=config.alarm_merge_gap_minutes,
                        alarm_persistence_windows=alarm_persistence_windows,
                    )
                    rows.append(
                        {
                            "model_variant": variant,
                            "test_subject": test_case,
                            "validation_subject": validation,
                            "train_subjects": ",".join(train),
                            "horizon_minutes": int(horizon),
                            "sph_minutes": config.sph_minutes,
                            "false_alarm_budget_per_hour": float(budget),
                            "alarm_persistence_windows": alarm_persistence_windows,
                            "validation_threshold": threshold,
                            "best_epoch": payload["best_epoch"],
                            "validation_mean_horizon_pr_auc": payload[
                                "best_validation_mean_horizon_pr_auc"
                            ],
                            **state_metrics,
                            **_window_metrics(labels, scores, threshold),
                            **event_metrics,
                        }
                    )
            del model
            print(f"Rebuilt checkpoint metrics [{run}/{total}]: {variant}/{test_case}", flush=True)
            incremental = pd.DataFrame(rows)
            temporary = resumable_results_path.with_suffix(resumable_results_path.suffix + ".partial")
            incremental.to_csv(temporary, index=False)
            os.replace(temporary, resumable_results_path)

    results = pd.DataFrame(rows)
    expected_rows = len(cases) * len(variants) * len(config.horizons_minutes) * len(config.far_budgets_per_hour)
    if len(results) != expected_rows:
        raise RuntimeError(f"Rebuilt {len(results)} rows, expected {expected_rows}.")
    summary = _aggregate(results)
    if len(summary) != len(variants) * len(config.horizons_minutes) * len(config.far_budgets_per_hour):
        raise RuntimeError("The rebuilt aggregate is incomplete.")

    for filename, frame in (
        (f"multihorizon_loso_per_patient{output_suffix}.csv", results),
        (f"multihorizon_loso_summary{output_suffix}.csv", summary),
    ):
        destination = output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".partial")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)

    protocol_manifest = {
        "protocol": "rebuild saved LOSO checkpoint predictions; no model retraining",
        "cases": cases,
        "unique_people": len(set("chb01_chb21" if case in {"chb01", "chb21"} else case for case in cases)),
        "model_variants": variants,
        "horizons_minutes": list(config.horizons_minutes),
        "sph_minutes": config.sph_minutes,
        "false_alarm_budgets_per_hour": list(config.far_budgets_per_hour),
        "alarm_merge_gap_minutes": config.alarm_merge_gap_minutes,
        "alarm_persistence_windows": alarm_persistence_windows,
        "threshold_selection": "Validation patient only; maximize event sensitivity subject to the selected FAR budget with the same alarm-persistence rule.",
        "test_labels_used_for_threshold_selection": False,
        "output_suffix": output_suffix,
    }
    manifest_path = output_dir / f"multihorizon_manifest{output_suffix}.json"
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    temporary_manifest.write_text(json.dumps(protocol_manifest, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)

    explanation_path = output_dir / "explanation_audit_per_event.csv"
    if explanation_path.is_file():
        explanation = pd.read_csv(explanation_path)
        if len(explanation) and explanation["patient_id"].nunique() == len(cases):
            audit_metrics = (
                "channel_rank_spearman_mean",
                "top_channel_jaccard_mean",
                "top_channel_mask_risk_drop_mean",
                "random_channel_mask_risk_drop_mean",
                "top_channel_mask_advantage",
            )
            audit_rows = []
            for horizon, group in explanation.groupby("horizon_minutes"):
                audit_row: dict[str, Any] = {
                    "horizon_minutes": int(horizon),
                    "patients": int(group["patient_id"].nunique()),
                    "events": int(group["event_index_in_recording"].count()),
                }
                for metric in audit_metrics:
                    values = group[metric].to_numpy(dtype=float)
                    valid = np.isfinite(values)
                    audit_row[f"{metric}_mean"] = float(np.nanmean(values)) if valid.any() else np.nan
                    audit_row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if valid.sum() > 1 else np.nan
                audit_rows.append(audit_row)
            pd.DataFrame(audit_rows).to_csv(
                output_dir / "explanation_audit_summary.csv", index=False
            )
    print(f"Atomically rebuilt {len(results)} rows and {len(summary)} aggregate groups.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results-multihorizon"))
    parser.add_argument("--samples-per-epoch", type=int, default=6000)
    parser.add_argument("--alarm-persistence-windows", type=int, default=1)
    parser.add_argument("--output-suffix", default="", help="Suffix for output CSV filenames; use to preserve prior results.")
    parser.add_argument("--variants", nargs="+", choices=MULTIHORIZON_MODEL_VARIANTS, default=list(MULTIHORIZON_MODEL_VARIANTS))
    args = parser.parse_args()
    rebuild(
        args.cache_dir,
        args.output_dir,
        args.samples_per_epoch,
        args.alarm_persistence_windows,
        args.output_suffix,
        args.variants,
    )


if __name__ == "__main__":
    main()
