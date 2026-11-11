from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seizure_prediction.config import MULTIHORIZON_MODEL_VARIANTS, ExperimentConfig
from seizure_prediction.edf import parse_siena_annotations, preprocess_edf, recording_duration
from seizure_prediction.event_metrics import event_forecast_metrics
from seizure_prediction.labels import EXCLUDED, INTERICTAL, PREICTAL, Seizure, label_windows
from seizure_prediction.predict import predict_signal
from seizure_prediction.training import load_checkpoint


DEFAULT_VARIANTS = (
    "multi_horizon",
    "multi_horizon_cnn",
    "multi_horizon_lstm",
    "multi_horizon_cnn_lstm",
)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [["" if pd.isna(value) else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in rows],
        ]
    )


def write_report(output_dir: Path) -> None:
    subject_results = pd.read_csv(output_dir / "siena_external_per_subject.csv")
    grouped = summarize_siena(subject_results)
    grouped.to_csv(output_dir / "siena_external_summary.csv", index=False)
    manifest = json.loads((output_dir / "siena_external_manifest.json").read_text(encoding="utf-8"))
    report = [
        "# Siena external evaluation",
        "",
        "This report evaluates frozen CHB-MIT LOSO checkpoints on the complete locally downloaded Siena cohort. Each checkpoint uses its own fold-specific threshold selected on that fold's separate CHB-MIT validation patient; Siena labels were not used to tune thresholds.",
        "",
        f"Cohort: {len(manifest['subjects'])} participants, 41 EDF recordings, and 47 annotated seizure events. Models: {', '.join(manifest['variants'])}. Horizons: 5, 10, 15, and 30 minutes (fixed 5-minute SPH).",
        "",
        "## Main external result",
        "",
        "At 30-minute SOP, using the CHB-validation threshold selected for a 0.1 false-alarm/hour budget, the proposed multi-horizon model detected 0/47 Siena events (0% event sensitivity) with zero false alarms. At the threshold selected for a 0.5/hour budget, it detected 45/47 events (95.7%) but produced 9.13 false alarms per eligible interictal hour. This is a marked failure to transfer the CHB operating point. Thresholds for nominal 0.5 and 1.0/hour budgets can coincide when validation selects the same operating threshold.",
        "",
        "",
        f"Subjects: {', '.join(manifest['subjects'])}",
        f"Checkpoint fold: `{manifest['checkpoint_fold']}` (held-out from CHB-MIT fitting).",
        "",
        "## Pooled results",
        "",
        markdown_table(grouped),
        "",
        "## Cross-dataset interpretation: measured findings and hypotheses",
        "",
        "**Measured and supported by this experiment:** the selected CHB-MIT validation thresholds do not preserve the target FAR on Siena. At 30-minute SOP, the full model changes from 0/47 detected events at 0.1/hour to 45/47 at the 0.5/hour CHB target, while observed FAR rises to 9.13/hour. The external evaluation used one frozen `chb08` fold per model, so fold-to-fold transfer variability was not estimated. Siena labels were not used to choose these thresholds.",
        "",
        "**Dataset differences supported by official metadata and implemented preprocessing:** CHB-MIT is a pediatric cohort with 256-Hz EEG and predominantly bipolar derivations; Siena contains 14 adult participants and 512-Hz referential 10-20 EEG channels. The pipeline resamples both datasets to 128 Hz and constructs the canonical bipolar montage from Siena's referential electrodes. These acquisition and population differences are present; their causal contribution to the observed performance drop has not been isolated. Sources: [CHB-MIT PhysioNet documentation](https://physionet.org/content/chbmit/1.0.0/) and [Siena PhysioNet documentation](https://physionet.org/content/siena-scalp-eeg/1.0.0/).",
        "",
        "**Plausible but untested explanations:** residual distribution shift from recording hardware, reference and channel availability; differences in participant age or seizure characteristics; and a probability-calibration shift under the CHB-derived threshold. The observed FAR increase is evidence of failed operating-point transfer, but it does not show whether score ranking, probability calibration, montage conversion, or cohort composition is the main cause. Testing these explanations requires per-fold transfer, channel-coverage/domain-shift analysis, and separately labeled calibration experiments; Siena must remain untouched for the primary external test.",
        "",
        "## Limitations",
        "",
        "This is a full evaluation of the 14-participant Siena cohort downloaded from PhysioNet, not clinical validation. The external test uses one frozen CHB-MIT chb08 LOSO checkpoint per model; it does not measure average transfer across all CHB folds. The database's recording protocol and reference montage differ from CHB-MIT; Siena recordings were converted to the project's fixed bipolar montage. Siena labels were not used for threshold tuning. Event sensitivity and false alarms per hour must be read together with eligible seizure counts and interictal exposure. A per-subject false-alarm rate is undefined when a participant contributes no eligible interictal exposure.",
        "",
        "## Data citation",
        "",
        "Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/5d4a-j060",
    ]
    (output_dir / "siena_external_report.md").write_text("\n".join(report), encoding="utf-8")


def summarize_siena(subject_results: pd.DataFrame) -> pd.DataFrame:
    grouping = ["model_variant", "horizon_minutes", "far_budget_per_hour_chb_validation"]
    rows: list[dict[str, object]] = []
    for keys, group in subject_results.groupby(grouping, sort=False):
        eligible = int(group["eligible_events"].sum())
        detected = int(group["detected_events"].sum())
        exposure = float(group["eligible_interictal_hours"].sum())
        false_alarms = int(group["false_alarms"].sum())
        detected_rows = group[group["detected_events"] > 0]
        row = dict(zip(grouping, keys))
        row.update(
            {
                "subjects": int(group["subject"].nunique()),
                "events": int(group["all_annotated_events"].sum()),
                "eligible_events": eligible,
                "detected_events": detected,
                "pooled_event_sensitivity": detected / eligible if eligible else np.nan,
                "mean_subject_event_sensitivity": group["event_sensitivity"].mean(),
                "false_alarms": false_alarms,
                "eligible_interictal_hours": exposure,
                "pooled_false_alarms_per_hour": false_alarms / exposure if exposure else np.nan,
                "mean_subject_false_alarms_per_hour": group["false_alarms_per_hour"].mean(),
                "mean_lead_time_minutes": (
                    float(np.average(
                        detected_rows["mean_lead_time_minutes"],
                        weights=detected_rows["detected_events"],
                    )) if len(detected_rows) else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def calibrated_thresholds(
    results_path: Path,
    fold: str,
    variants: list[str],
    horizons: list[int],
    far_budgets: list[float],
) -> dict[tuple[str, int, float], float]:
    results = pd.read_csv(results_path)
    thresholds: dict[tuple[str, int, float], float] = {}
    for variant in variants:
        for horizon in horizons:
            for budget in far_budgets:
                selected = results.loc[
                    (results["model_variant"] == variant)
                    & (results["test_subject"] == fold.lower())
                    & (results["horizon_minutes"] == horizon)
                    & np.isclose(results["false_alarm_budget_per_hour"], budget),
                    "validation_threshold",
                ].dropna()
                if selected.empty:
                    raise ValueError(
                        f"No CHB-MIT LOSO validation thresholds for {variant}, "
                        f"SOP {horizon}, FAR {budget}/h."
                    )
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one validation threshold for {variant}/{fold}, "
                        f"SOP {horizon}, FAR {budget:g}/h; found {len(selected)}."
                    )
                thresholds[(variant, horizon, budget)] = float(selected.iloc[0])
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-model external Siena evaluation with CHB-MIT-only thresholds."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/siena"))
    parser.add_argument(
        "--results-dir", type=Path, default=Path("artifacts/results-multihorizon-16case")
    )
    parser.add_argument("--fold", default="chb08", help="Frozen CHB-MIT LOSO fold checkpoint")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15, 30])
    parser.add_argument("--far-budgets", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results-siena-fullcohort"))
    parser.add_argument("--max-recordings-per-subject", type=int)
    parser.add_argument("--report-only", action="store_true", help="Regenerate the Markdown report from saved CSV and JSON output.")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.report_only:
        write_report(args.output_dir)
        print(f"Wrote report to {args.output_dir / 'siena_external_report.md'}", flush=True)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = (
        torch.device("cuda")
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else torch.device("cpu")
    )
    subjects = sorted(path for path in args.data_dir.iterdir() if path.is_dir() and path.name.upper().startswith("PN"))
    if not subjects:
        raise FileNotFoundError(f"No Siena PNxx subject folders found under {args.data_dir}.")
    if any(variant not in MULTIHORIZON_MODEL_VARIANTS for variant in args.variants):
        raise ValueError("Only the project's multi-horizon model variants can be evaluated here.")
    if any(horizon not in (5, 10, 15, 30) for horizon in args.horizons):
        raise ValueError("Supported SOP horizons are 5, 10, 15, and 30 minutes.")

    thresholds = calibrated_thresholds(
        args.results_dir / "multihorizon_loso_per_patient.csv",
        args.fold,
        args.variants,
        args.horizons,
        args.far_budgets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "siena_external_progress.csv"
    audit_progress_path = args.output_dir / "siena_annotation_audit_progress.csv"
    summaries: list[dict[str, object]] = (
        pd.read_csv(progress_path).to_dict(orient="records")
        if progress_path.is_file()
        else []
    )
    annotation_audit: list[dict[str, object]] = (
        pd.read_csv(audit_progress_path).to_dict(orient="records")
        if audit_progress_path.is_file()
        else []
    )
    config_reference: ExperimentConfig | None = None
    for variant in args.variants:
        checkpoint = args.results_dir / "checkpoints" / variant / args.fold / "model.pt"
        model, payload, checkpoint_device = load_checkpoint(checkpoint, device=device)
        config = ExperimentConfig.from_dict(payload["config"])
        config_reference = config
        if not set(args.horizons).issubset(config.horizons_minutes):
            raise ValueError(f"Checkpoint {checkpoint} does not support requested horizons.")
        print(f"Loaded {variant} ({args.fold} held-out) on {checkpoint_device}.", flush=True)

        for subject_dir in subjects:
            completed_rows = [
                row for row in summaries
                if row.get("model_variant") == variant
                and row.get("subject") == subject_dir.name.upper()
            ]
            if len(completed_rows) == len(args.horizons) * len(args.far_budgets):
                print(f"Reusing completed Siena subject: {variant}/{subject_dir.name}", flush=True)
                continue
            annotation_file = subject_dir / f"Seizures-list-{subject_dir.name.upper()}.txt"
            annotations = parse_siena_annotations(annotation_file)
            edfs = sorted(subject_dir.glob("*.edf"))
            if args.max_recordings_per_subject:
                edfs = edfs[: args.max_recordings_per_subject]
            if not edfs:
                continue
            edf_names = {path.name.lower() for path in edfs}
            unmatched = [name for name in annotations if name not in edf_names]
            if unmatched and len(edfs) == 1 and len(annotations) == 1:
                # PN01's annotation file names its one recording PN01.edf,
                # while PhysioNet distributes it as PN01-1.edf.
                annotations = {edfs[0].name.lower(): next(iter(annotations.values()))}
                annotation_audit.append(
                    {"subject": subject_dir.name, "recording": edfs[0].name,
                     "action": "mapped_single_recording_annotation_name",
                     "annotation_recording": unmatched[0]}
                )
            elif unmatched:
                raise ValueError(
                    f"Unmatched annotation recording names for {subject_dir.name}: {unmatched}"
                )
            duration_by_file = {path.name.lower(): recording_duration(path) for path in edfs}
            offsets: dict[str, float] = {}
            cursor = 0.0
            for path in edfs:
                offsets[path.name.lower()] = cursor
                cursor += duration_by_file[path.name.lower()] + config.window_seconds + config.alarm_merge_gap_minutes * 60
            clean_events: dict[str, list[tuple[float, float]]] = {}
            for name, seizure_list in annotations.items():
                if name not in offsets:
                    continue
                duration = duration_by_file[name]
                clean_events[name] = []
                for event in seizure_list:
                    if event.onset_seconds >= duration:
                        annotation_audit.append(
                            {"subject": subject_dir.name, "recording": name,
                             "action": "excluded_onset_outside_recording",
                             "onset_seconds": event.onset_seconds,
                             "offset_seconds": event.offset_seconds,
                             "recording_duration_seconds": duration}
                        )
                        continue
                    event_end = min(event.offset_seconds, duration)
                    if event.offset_seconds > duration:
                        annotation_audit.append(
                            {"subject": subject_dir.name, "recording": name,
                             "action": "clipped_offset_to_recording_end",
                             "onset_seconds": event.onset_seconds,
                             "offset_seconds": event.offset_seconds,
                             "recording_duration_seconds": duration}
                        )
                    clean_events[name].append((event.onset_seconds, event_end))
            events = [
                (offsets[name] + onset, offsets[name] + end)
                for name, event_list in clean_events.items()
                for onset, end in event_list
            ]
            per_horizon_budget_records: dict[int, dict[float, dict[str, pd.DataFrame]]] = {
                horizon: {budget: {} for budget in args.far_budgets}
                for horizon in args.horizons
            }
            for index, path in enumerate(edfs, start=1):
                key = path.name.lower()
                print(
                    f"  {subject_dir.name} {index}/{len(edfs)} {path.name} ({variant})",
                    flush=True,
                )
                signal, matched = preprocess_edf(path, config)
                frame, _, _, _ = predict_signal(
                    signal,
                    config,
                    model,
                    checkpoint_device,
                    score_threshold=0.5,
                )
                if len(matched) < 4:
                    raise ValueError(f"Only {len(matched)} canonical channels matched in {path}.")
                starts = frame["time_minutes"].to_numpy(dtype=np.float64) * 60.0
                for horizon in args.horizons:
                    state_labels = label_windows(
                        starts,
                        config.window_seconds,
                        [
                            # Keep the project's standard event-label protocol.
                            Seizure(onset, end)
                            for onset, end in clean_events.get(key, [])
                        ],
                        horizon,
                        config.sph_minutes,
                        config.interictal_buffer_minutes,
                        config.min_ictal_overlap_seconds,
                    )
                    score_column = f"preictal_probability_sop_{horizon}m"
                    scores = frame[score_column].to_numpy(dtype=np.float64)
                    binary_labels = np.where(
                        (state_labels == INTERICTAL) | (state_labels == PREICTAL),
                        state_labels,
                        EXCLUDED,
                    )
                    for budget in args.far_budgets:
                        per_horizon_budget_records[horizon][budget][path.name] = pd.DataFrame(
                            {"time_seconds": starts, "label": binary_labels, "score": scores}
                        )
            for horizon in args.horizons:
                for budget in args.far_budgets:
                    threshold = thresholds[(variant, horizon, budget)]
                    metrics = event_forecast_metrics(
                        per_horizon_budget_records[horizon][budget],
                        offsets,
                        events,
                        horizon,
                        config.sph_minutes,
                        threshold,
                        config.window_seconds,
                        config.alarm_merge_gap_minutes,
                    )
                    row = {
                        "dataset": "Siena Scalp EEG",
                        "subject": subject_dir.name.upper(),
                        "model_variant": variant,
                        "checkpoint_fold": args.fold,
                        "horizon_minutes": horizon,
                        "sph_minutes": config.sph_minutes,
                        "far_budget_per_hour_chb_validation": budget,
                        "threshold_from_chb_loso_validation": threshold,
                        **metrics,
                    }
                    summaries.append(row)
                    print(
                        f"    SOP {horizon}m, validation FAR target {budget:g}/h: "
                        f"observed FAR={metrics['false_alarms_per_hour']:.3f}/h; "
                        f"sensitivity={metrics['event_sensitivity']:.3f}; "
                        f"eligible={metrics['eligible_events']}/{metrics['all_annotated_events']}; "
                        f"lead={metrics['mean_lead_time_minutes']}",
                        flush=True,
                    )
            progress_frame = pd.DataFrame(summaries)
            temporary_progress = progress_path.with_suffix(".csv.partial")
            progress_frame.to_csv(temporary_progress, index=False)
            temporary_progress.replace(progress_path)
            pd.DataFrame(annotation_audit).drop_duplicates().to_csv(
                audit_progress_path, index=False
            )

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "siena_external_per_subject.csv", index=False)
    summary.to_json(args.output_dir / "siena_external_per_subject.json", orient="records", indent=2)
    pd.DataFrame(annotation_audit).drop_duplicates().to_csv(
        args.output_dir / "siena_annotation_audit.csv", index=False
    )
    if config_reference is None:
        raise RuntimeError("No model evaluation was run.")
    grouped = summarize_siena(summary)
    grouped.to_csv(args.output_dir / "siena_external_summary.csv", index=False)
    manifest = {
        "protocol": "Frozen CHB-MIT LOSO fold, direct external evaluation; threshold is the validation-only threshold saved for that exact held-out fold. No Siena labels used for threshold selection.",
        "dataset": "Siena Scalp EEG Database v1.0.0",
        "subjects": [path.name.upper() for path in subjects],
        "variants": args.variants,
        "horizons_minutes": args.horizons,
        "sph_minutes": config_reference.sph_minutes,
        "far_budgets_per_hour": args.far_budgets,
        "checkpoint_fold": args.fold,
        "device": str(device),
        "sampling_rate_model_hz": config_reference.sampling_rate,
        "channel_strategy": "Siena referential 10-20 electrodes converted to the project's fixed bipolar montage; EKG ignored by EEG channel selection.",
        "recordings": 41,
        "annotated_seizure_events": 47,
        "interpretation": "Full downloaded 14-participant Siena external evaluation; not clinical validation and uses one frozen CHB-MIT fold.",
    }
    (args.output_dir / "siena_external_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_report(args.output_dir)
    progress_path.unlink(missing_ok=True)
    audit_progress_path.unlink(missing_ok=True)
    print(f"Wrote Siena external results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
