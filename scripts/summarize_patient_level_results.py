"""Create unique-person summaries from the CHB-MIT case-level LOSO table.

CHB-MIT cases chb01 and chb21 are two recordings/cases for the same person.
This script combines their held-out case metrics before computing across-person
mean and sample SD, so patient-level summaries do not count that person twice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CASE_TO_PERSON = {"chb01": "chb01_chb21", "chb21": "chb01_chb21"}
SUM_METRICS = {
    "window_sensitivity",
    "window_specificity",
    "window_f1",
    "window_pr_auc",
    "state_accuracy",
    "state_macro_f1",
}


def person_row(group: pd.DataFrame) -> pd.Series:
    row = group.iloc[0].copy()
    row["test_person"] = CASE_TO_PERSON.get(row.test_subject, row.test_subject)
    row["test_cases"] = len(group)
    row["test_case_ids"] = ",".join(sorted(group.test_subject.astype(str)))

    eligible_events = group.eligible_events.sum()
    detected_events = group.detected_events.sum()
    interictal_hours = group.eligible_interictal_hours.sum()
    false_alarms = group.false_alarms.sum()
    row["eligible_events"] = eligible_events
    row["detected_events"] = detected_events
    row["all_annotated_events"] = group.all_annotated_events.sum()
    row["event_sensitivity"] = (
        detected_events / eligible_events if eligible_events else np.nan
    )
    row["eligible_interictal_hours"] = interictal_hours
    row["false_alarms"] = false_alarms
    row["false_alarms_per_hour"] = (
        false_alarms / interictal_hours if interictal_hours else np.nan
    )
    row["alarm_count"] = group.alarm_count.sum()

    detected_rows = group[group.detected_events > 0]
    if len(detected_rows):
        row["mean_lead_time_minutes"] = np.average(
            detected_rows.mean_lead_time_minutes,
            weights=detected_rows.detected_events,
        )
        row["median_lead_time_minutes"] = np.average(
            detected_rows.median_lead_time_minutes,
            weights=detected_rows.detected_events,
        )
    else:
        row["mean_lead_time_minutes"] = np.nan
        row["median_lead_time_minutes"] = np.nan

    # Window metrics are retained as the unweighted mean of the component case
    # metrics for a multi-case person; raw window confusion counts were not
    # retained by the original evaluator, so a sample-weighted recomputation is
    # not possible. Single-case people retain their exact case value.
    for metric in SUM_METRICS:
        row[metric] = group[metric].mean()
    row["test_subject"] = row["test_person"]
    return row


def build_patient_level_summaries(
    results_dir: Path, input_suffix: str = "", output_suffix: str = ""
) -> tuple[Path, Path, int, int]:
    if input_suffix and not input_suffix.startswith("_"):
        input_suffix = f"_{input_suffix}"
    if output_suffix and not output_suffix.startswith("_"):
        output_suffix = f"_{output_suffix}"
    source = results_dir / f"multihorizon_loso_per_patient{input_suffix}.csv"
    per_person_path = results_dir / f"multihorizon_loso_per_person{output_suffix}.csv"
    summary_path = results_dir / f"multihorizon_loso_person_summary{output_suffix}.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Case-level LOSO table not found: {source}")

    cases = pd.read_csv(source)
    cases["test_person"] = cases.test_subject.map(CASE_TO_PERSON).fillna(cases.test_subject)
    keys = ["model_variant", "test_person", "horizon_minutes", "sph_minutes", "false_alarm_budget_per_hour"]
    if "alarm_persistence_windows" in cases.columns:
        keys.append("alarm_persistence_windows")
    person_rows = []
    for values, group in cases.groupby(keys, sort=False, dropna=False):
        row = person_row(group)
        for key, value in zip(keys, values):
            row[key] = value
        person_rows.append(row)
    people = pd.DataFrame(person_rows)
    people.to_csv(per_person_path, index=False)

    rows = []
    summary_metrics = [
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
    ]
    grouping = ["model_variant", "horizon_minutes", "false_alarm_budget_per_hour"]
    if "alarm_persistence_windows" in people.columns:
        grouping.append("alarm_persistence_windows")
    for values, group in people.groupby(grouping, sort=False, dropna=False):
        row = dict(zip(grouping, values))
        row["people"] = group.test_person.nunique()
        for metric in summary_metrics:
            valid = group[metric].dropna()
            row[f"{metric}_mean"] = valid.mean() if len(valid) else np.nan
            row[f"{metric}_std"] = valid.std(ddof=1) if len(valid) > 1 else np.nan
            row[f"{metric}_n"] = len(valid)
        budget = row["false_alarm_budget_per_hour"]
        row["people_meeting_far_target"] = int(
            (group.false_alarms_per_hour <= budget).sum()
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return per_person_path, summary_path, len(people), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--input-suffix", default="")
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    per_person_path, summary_path, people_rows, summary_rows = build_patient_level_summaries(
        args.results_dir, args.input_suffix, args.output_suffix
    )
    print(f"Wrote {people_rows:,} person/model/horizon/budget rows to {per_person_path}")
    print(f"Wrote {summary_rows:,} unique-person summary rows to {summary_path}")


if __name__ == "__main__":
    main()
