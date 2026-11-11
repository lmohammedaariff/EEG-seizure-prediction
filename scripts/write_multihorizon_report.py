"""Create a reproducible human-readable report from completed LOSO CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_patient_level_results import build_patient_level_summaries
from paired_bootstrap_analysis import analyze as analyze_paired_bootstrap


EXPECTED_MODELS = (
    "multi_horizon_temporal",
    "multi_horizon_temporal_spectral",
    "multi_horizon_multi_domain",
    "multi_horizon",
    "multi_horizon_cnn",
    "multi_horizon_lstm",
    "multi_horizon_cnn_lstm",
)
EXPECTED_HORIZONS = (5, 10, 15, 30)
EXPECTED_BUDGETS = (0.1, 0.5, 1.0)


def _fmt(mean: object, std: object, percent: bool = False) -> str:
    if pd.isna(mean):
        return "n/a"
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    std_text = f"{float(std) * scale:.1f}{suffix}" if not pd.isna(std) else "n/a"
    return f"{float(mean) * scale:.1f}{suffix} ± {std_text}"


def build_report(results_dir: Path, destination: Path) -> None:
    summary_path = results_dir / "multihorizon_loso_summary.csv"
    patients_path = results_dir / "multihorizon_loso_per_patient.csv"
    explanation_path = results_dir / "explanation_audit_summary.csv"
    labels_path = results_dir / "label_audit.csv"
    missing = [path.name for path in (summary_path, patients_path, labels_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed experiment outputs: {', '.join(missing)}")

    summary = pd.read_csv(summary_path)
    patients = pd.read_csv(patients_path)
    labels = pd.read_csv(labels_path)
    _, person_summary_path, _, _ = build_patient_level_summaries(results_dir)
    person_summary = pd.read_csv(person_summary_path)
    person_rows_path = results_dir / "multihorizon_loso_per_person.csv"
    if not person_rows_path.is_file():
        raise FileNotFoundError(f"Missing unique-person detail table: {person_rows_path}")
    person_rows = pd.read_csv(person_rows_path)
    case_ids = sorted(patients["test_subject"].dropna().astype(str).unique())
    case_count = len(case_ids)
    individual_count = case_count - int({"chb01", "chb21"}.issubset(case_ids))
    expected = {
        (model, horizon, budget)
        for model in EXPECTED_MODELS
        for horizon in EXPECTED_HORIZONS
        for budget in EXPECTED_BUDGETS
    }
    actual = {
        (row.model_variant, int(row.horizon_minutes), float(row.false_alarm_budget_per_hour))
        for row in summary.itertuples()
    }
    if actual != expected or summary["subjects"].min() != case_count:
        raise ValueError(
            "The report requires all 7 models × 4 horizons × 3 budgets with results for every held-out case."
        )

    rows = [
        "# Multi-horizon CHB-MIT LOSO results",
        "",
        "## Study protocol",
        "",
        f"The report aggregates {case_count} included CHB-MIT case folders ({individual_count} distinct people): {', '.join(f'`{case}`' for case in case_ids)}. Each test case is excluded from fitting and threshold selection; a separate participant group is reserved for validation. Cases `chb01` and `chb21` are never split across train/validation/test and are combined for person-level summaries.",
        "",
        "A shared multi-horizon model predicts SOPs of 5, 10, 15, and 30 minutes with a fixed 5-minute SPH. Thresholds are selected on validation data to maximize event sensitivity under 0.1, 0.5, and 1.0 false alarms per interictal hour. These budgets are validation targets; actual held-out test rates can differ. Reported event sensitivity, false alarms/hour, and lead time are event-level outcomes. Window sensitivity and PR-AUC are separate window-level diagnostics.",
        "",
        "## Results",
        "",
        "The detailed table below shows case-level means ± sample SD. A unique-person headline is added after the table; it combines the paired `chb01`/`chb21` cases before averaging across people. Valid n shows the denominator for event sensitivity, false alarms/hour, lead time, window sensitivity, and window PR-AUC. Event sensitivity and window sensitivity are percentages; lead time is minutes.",
        "",
        "| Model | SOP (min) | FAR target (/h) | Valid n (event/FAR/lead/window sens/PR-AUC) | Test cases meeting FAR target | Event sensitivity (%) | FAR (/h) | Lead time (min) | Window sensitivity (%) | Window PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = (
        ("event_sensitivity", True),
        ("false_alarms_per_hour", False),
        ("mean_lead_time_minutes", False),
        ("window_sensitivity", True),
        ("window_pr_auc", False),
    )
    for item in summary.sort_values(
        ["horizon_minutes", "false_alarm_budget_per_hour", "model_variant"]
    ).itertuples():
        group = patients.loc[
            (patients["model_variant"] == item.model_variant)
            & (patients["horizon_minutes"] == item.horizon_minutes)
            & (patients["false_alarm_budget_per_hour"] == item.false_alarm_budget_per_hour)
        ]
        valid_counts = "/".join(
            str(int(np.isfinite(group[metric].to_numpy(dtype=float)).sum()))
            for metric, _ in metrics
        )
        far_values = group["false_alarms_per_hour"].to_numpy(dtype=float)
        budget_met = int(np.sum(np.isfinite(far_values) & (far_values <= item.false_alarm_budget_per_hour)))
        values = [
            _fmt(getattr(item, f"{metric}_mean"), getattr(item, f"{metric}_std"), percent)
            for metric, percent in metrics
        ]
        rows.append(
            f"| {item.model_variant} | {int(item.horizon_minutes)} | {item.false_alarm_budget_per_hour:g} | {valid_counts} | {budget_met}/{case_count} | "
            + " | ".join(values)
            + " |"
        )

    rows.extend(["", "## Evaluable seizure denominators", ""])
    rows.append(
        "The full multi-horizon model (`multi_horizon`) was evaluated on 100 annotated seizures across 15 unique people. Under the current SOP/SPH label and recording-coverage rule, 98 seizures had at least one fully labeled forecast window and were evaluable at each horizon; two annotated events lacked such a window. Event counts are not independent sample units; patient-level inference uses 15 people. The table reports pooled detections at the 0.5/hour validation-target operating point; observed held-out FAR may exceed that target."
    )
    rows.extend([
        "",
        "| SOP (min) | Unique people | Annotated seizures | Evaluable seizures | Detected / evaluable at 0.5/h target |",
        "|---:|---:|---:|---:|---:|",
    ])
    denominator_rows = person_rows.loc[
        (person_rows.model_variant == "multi_horizon")
        & (person_rows.false_alarm_budget_per_hour == 0.5)
    ]
    for horizon, group in denominator_rows.groupby("horizon_minutes", sort=True):
        rows.append(
            f"| {int(horizon)} | {group.test_person.nunique()} | {int(group.all_annotated_events.sum())} | {int(group.eligible_events.sum())} | {int(group.detected_events.sum())}/{int(group.eligible_events.sum())} |"
        )

    rows.extend([
        "",
        "The 84-row comparison table above covers all seven variants (four ablations and three matched baselines), four SOPs, and three validation FAR targets. Fold-level metrics and per-fold evaluable/detected seizure counts are in [`multihorizon_loso_per_patient.csv`](multihorizon_loso_per_patient.csv); the unique-person table is in [`multihorizon_loso_per_person.csv`](multihorizon_loso_per_person.csv).",
        "",
        "## Unique-person results",
        "",
    ])
    primary = person_summary.loc[
        (person_summary.model_variant == "multi_horizon")
        & (person_summary.horizon_minutes == 30)
    ].set_index("false_alarm_budget_per_hour")
    for budget in EXPECTED_BUDGETS:
        item = primary.loc[budget]
        rows.append(
            f"At the 30-minute SOP and {budget:g}/hour validation target, the full multi-domain, subject-adversarial model (`multi_horizon`) achieved "
            f"event sensitivity **{_fmt(item.event_sensitivity_mean, item.event_sensitivity_std, True)}**, "
            f"observed FAR **{_fmt(item.false_alarms_per_hour_mean, item.false_alarms_per_hour_std)}/hour**, "
            f"and mean warning lead **{_fmt(item.mean_lead_time_minutes_mean, item.mean_lead_time_minutes_std)} minutes**; "
            f"{int(item.people_meeting_far_target)} of {int(item.people)} people met the FAR target."
        )
    baseline = person_summary.loc[
        (person_summary.model_variant == "multi_horizon_lstm")
        & (person_summary.horizon_minutes == 30)
        & (person_summary.false_alarm_budget_per_hour == 1.0)
    ].iloc[0]
    rows.extend(
        [
            "",
            f"At the 30-minute SOP and 1.0/hour target, the matched LSTM baseline achieved event sensitivity **{_fmt(baseline.event_sensitivity_mean, baseline.event_sensitivity_std, True)}** and observed FAR **{_fmt(baseline.false_alarms_per_hour_mean, baseline.false_alarms_per_hour_std)}/hour**, with {int(baseline.people_meeting_far_target)} of {int(baseline.people)} people meeting the target. The model variants trade sensitivity against false alarms; no single variant dominates every operating point.",
        ]
    )

    rows.extend(
        [
            "",
            "## Label and imbalance audit",
            "",
            f"The label audit contains {len(labels)} patient-by-horizon rows. Preictal-to-interictal window ratios ranged from {labels['preictal_to_interictal_ratio'].min():.4f} to {labels['preictal_to_interictal_ratio'].max():.4f}. Training uses patient/state-balanced sampling and horizon-specific loss weights estimated from training patients only. Ictal and ambiguous windows are excluded from the binary preictal-versus-interictal target.",
            "",
            "## Explanation audit",
            "",
        ]
    )
    if explanation_path.is_file():
        explanation = pd.read_csv(explanation_path)
        if explanation.empty:
            rows.append("No event explanation audit rows were produced.")
        else:
            rows.extend(
                [
                    "Integrated Gradients channel rankings are evaluated for repeatability across preictal event windows. Top-attributed-channel masking is compared with random-channel masking of the same size; these are stability and perturbation-faithfulness checks, not clinical validation.",
                    "",
                    "| SOP (min) | Patients | Events | Channel rank Spearman | Top-channel Jaccard | Top-mask risk drop | Random-mask risk drop | Top-mask advantage |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            explanation_columns = (
                ("channel_rank_spearman_mean_mean", "channel_rank_spearman_mean_std"),
                ("top_channel_jaccard_mean_mean", "top_channel_jaccard_mean_std"),
                ("top_channel_mask_risk_drop_mean_mean", "top_channel_mask_risk_drop_mean_std"),
                ("random_channel_mask_risk_drop_mean_mean", "random_channel_mask_risk_drop_mean_std"),
                ("top_channel_mask_advantage_mean", "top_channel_mask_advantage_std"),
            )
            for item in explanation.sort_values("horizon_minutes").itertuples():
                fields = [
                    item.horizon_minutes,
                    item.patients,
                    item.events,
                    *(_fmt(getattr(item, mean), getattr(item, std)) for mean, std in explanation_columns),
                ]
                rows.append("| " + " | ".join(str(value) for value in fields) + " |")
    else:
        rows.append("Explanation audit summary is not present.")

    rows.extend(
        [
            "",
            "## Uncertainty and paired comparisons",
            "",
            "Person-cluster bootstrap confidence intervals and paired participant-level model contrasts are generated from the completed unique-person table. The resampling unit is the held-out person (with chb01/chb21 combined), rather than windows or seizures. Direction counts show how many of the paired people favor each model for event sensitivity and FAR; they are descriptive. With only 15 people, intervals are exploratory and do not establish equivalence or correct for multiple model/horizon/operating-point comparisons.",
            "",
            "See [`statistical_uncertainty_report.md`](statistical_uncertainty_report.md), [`person_bootstrap_confidence_intervals.csv`](person_bootstrap_confidence_intervals.csv), and [`paired_model_comparisons.csv`](paired_model_comparisons.csv).",
            "",
            "## Interpretation and limits",
            "",
            "A validation FAR budget does not cap the rate on unseen people. Use per-person observed FAR and the people-meeting-target count to assess alarm control; where rates exceed the target, this run shows a validation-to-test calibration gap and the sensitivity/FAR trade-off is not yet solved.",
            "",
            f"This is retrospective within-database evaluation on a limited pediatric cohort, not external validation or evidence of clinical utility. The multi-horizon, domain-fusion, subject-adversarial, CNN, LSTM, and CNN-LSTM components are established techniques; the defensible contribution is the controlled comparison and event-level operating-point protocol with explanation audits, not a claim that any single component is unprecedented. The {individual_count} distinct people provide limited statistical power, so report case-wise results alongside aggregate summaries.",
            "",
            "For unique-person mean ± SD, see [`multihorizon_loso_person_summary.csv`](multihorizon_loso_person_summary.csv); individual person-level values are in [`multihorizon_loso_per_person.csv`](multihorizon_loso_per_person.csv). Detailed held-out case rows remain in [`multihorizon_loso_per_patient.csv`](multihorizon_loso_per_patient.csv). For label counts and exclusions, see [`label_audit.csv`](label_audit.csv).",
            "",
        ]
    )
    analyze_paired_bootstrap(results_dir, draws=10000, seed=20261016)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/results-multihorizon"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results-multihorizon/final_results_report.md"))
    args = parser.parse_args()
    build_report(args.results_dir, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
