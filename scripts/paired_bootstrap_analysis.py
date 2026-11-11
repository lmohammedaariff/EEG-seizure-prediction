"""Patient-cluster bootstrap intervals and paired model contrasts for LOSO results.

The resampling unit is the unique held-out person, never an EEG window or seizure.
This preserves the person-level evaluation unit used by the primary study.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("event_sensitivity", "false_alarms_per_hour", "mean_lead_time_minutes")
REFERENCE = "multi_horizon"
COMPARATORS = (
    "multi_horizon_cnn",
    "multi_horizon_lstm",
    "multi_horizon_cnn_lstm",
    "multi_horizon_temporal",
    "multi_horizon_temporal_spectral",
    "multi_horizon_multi_domain",
)


def _bootstrap_mean_ci(
    values: np.ndarray, *, draws: int, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan, 0
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high), int(len(values))


def _paired_difference_ci(
    first: np.ndarray,
    second: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    valid = np.isfinite(first) & np.isfinite(second)
    differences = first[valid] - second[valid]
    if not len(differences):
        return np.nan, np.nan, np.nan, 0
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high), int(len(differences))


def analyze(results_dir: Path, draws: int, seed: int) -> tuple[Path, Path, Path]:
    source = results_dir / "multihorizon_loso_per_person.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Missing unique-person table: {source}")
    people = pd.read_csv(source)
    required = {
        "model_variant", "test_person", "horizon_minutes",
        "false_alarm_budget_per_hour", *METRICS,
    }
    missing = sorted(required - set(people.columns))
    if missing:
        raise ValueError(f"LOSO table is missing columns: {missing}")
    if people.duplicated(
        ["model_variant", "test_person", "horizon_minutes", "false_alarm_budget_per_hour"]
    ).any():
        raise ValueError("Expected one row per model/person/horizon/FAR target.")

    rng = np.random.default_rng(seed)
    interval_rows: list[dict[str, object]] = []
    compare_rows: list[dict[str, object]] = []
    for (horizon, budget), stratum in people.groupby(
        ["horizon_minutes", "false_alarm_budget_per_hour"], sort=True
    ):
        for model, group in stratum.groupby("model_variant", sort=True):
            for metric in METRICS:
                mean, low, high, n = _bootstrap_mean_ci(
                    group[metric].to_numpy(dtype=float), draws=draws, rng=rng
                )
                interval_rows.append({
                    "model_variant": model,
                    "horizon_minutes": int(horizon),
                    "false_alarm_budget_per_hour": float(budget),
                    "metric": metric,
                    "mean": mean,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "people_n": n,
                    "bootstrap_draws": draws,
                    "seed": seed,
                })

        reference = stratum[stratum.model_variant == REFERENCE].set_index("test_person")
        for comparator in COMPARATORS:
            other = stratum[stratum.model_variant == comparator].set_index("test_person")
            common = reference.index.intersection(other.index)
            for metric in ("event_sensitivity", "false_alarms_per_hour"):
                delta, low, high, n = _paired_difference_ci(
                    reference.loc[common, metric].to_numpy(dtype=float),
                    other.loc[common, metric].to_numpy(dtype=float),
                    draws=draws,
                    rng=rng,
                )
                first = reference.loc[common, metric].to_numpy(dtype=float)
                second = other.loc[common, metric].to_numpy(dtype=float)
                valid = np.isfinite(first) & np.isfinite(second)
                paired_delta = first[valid] - second[valid]
                if metric == "event_sensitivity":
                    reference_better = int(np.sum(paired_delta > 1e-12))
                    comparator_better = int(np.sum(paired_delta < -1e-12))
                else:
                    reference_better = int(np.sum(paired_delta < -1e-12))
                    comparator_better = int(np.sum(paired_delta > 1e-12))
                ties = int(np.sum(np.abs(paired_delta) <= 1e-12))
                compare_rows.append({
                    "reference_model": REFERENCE,
                    "comparator_model": comparator,
                    "horizon_minutes": int(horizon),
                    "false_alarm_budget_per_hour": float(budget),
                    "metric": metric,
                    "paired_mean_difference_reference_minus_comparator": delta,
                    "bootstrap_95ci_low": low,
                    "bootstrap_95ci_high": high,
                    "paired_people_n": n,
                    "reference_better_people_n": reference_better,
                    "tie_people_n": ties,
                    "comparator_better_people_n": comparator_better,
                    "bootstrap_draws": draws,
                    "seed": seed,
                })

    intervals_path = results_dir / "person_bootstrap_confidence_intervals.csv"
    comparisons_path = results_dir / "paired_model_comparisons.csv"
    report_path = results_dir / "statistical_uncertainty_report.md"
    intervals = pd.DataFrame(interval_rows)
    comparisons = pd.DataFrame(compare_rows)
    intervals.to_csv(intervals_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)

    rows = [
        "# Statistical uncertainty and paired model comparisons",
        "",
        f"Results use {people.test_person.nunique()} unique held-out people. The `chb01`/`chb21` cases are combined before resampling.",
        "Bootstrap intervals resample people with replacement (not windows or seizures); paired model contrasts use the same resampled people for both models. Intervals are percentile 95% confidence intervals with a fixed seed. These exploratory intervals do not correct for multiple comparisons and are imprecise with this small cohort.",
        "",
        "A paired interval entirely above zero favors the shared multi-horizon model for event sensitivity, while for false alarms/hour a value below zero favors the shared model. Overlap with zero does not establish equivalence.",
        "",
        "## Shared multi-horizon model at SOP 30 minutes",
        "",
        "| Validation FAR target (/h) | Metric | Mean | Person-bootstrap 95% CI | n people |",
        "|---:|---|---:|---:|---:|",
    ]
    primary = intervals[
        (intervals.model_variant == REFERENCE)
        & (intervals.horizon_minutes == 30)
        & (intervals.metric.isin(("event_sensitivity", "false_alarms_per_hour")))
    ]
    for row in primary.sort_values(["false_alarm_budget_per_hour", "metric"]).itertuples():
        rows.append(
            f"| {row.false_alarm_budget_per_hour:g} | {row.metric} | {row.mean:.3f} | [{row.bootstrap_95ci_low:.3f}, {row.bootstrap_95ci_high:.3f}] | {row.people_n} |"
        )
    rows.extend([
        "",
        "## Paired SOP 30-minute model contrasts",
        "",
        "The table reports shared multi-horizon minus comparator using the same people for both models. Direction counts are descriptive: for sensitivity, better means higher; for FAR, better means lower. They do not replace the paired bootstrap interval or establish statistical significance.",
        "",
        "| FAR target (/h) | Comparator | Metric | Mean paired difference | Person-bootstrap 95% CI | Reference better / tie / comparator better (people) |",
        "|---:|---|---|---:|---:|---:|",
    ])
    paired = comparisons[
        (comparisons.horizon_minutes == 30)
        & (comparisons.false_alarm_budget_per_hour == 0.5)
        & (comparisons.metric.isin(("event_sensitivity", "false_alarms_per_hour")))
    ]
    for row in paired.sort_values(["comparator_model", "metric"]).itertuples():
        rows.append(
            f"| {row.false_alarm_budget_per_hour:g} | {row.comparator_model} | {row.metric} | {row.paired_mean_difference_reference_minus_comparator:.3f} | [{row.bootstrap_95ci_low:.3f}, {row.bootstrap_95ci_high:.3f}] | {row.reference_better_people_n} / {row.tie_people_n} / {row.comparator_better_people_n} |"
        )
    rows.extend([
        "",
        "Full interval and comparison tables are `person_bootstrap_confidence_intervals.csv` and `paired_model_comparisons.csv`. Thresholds remain those selected on the fold-specific validation person; the test participant is never used for threshold selection.",
        "",
    ])
    report_path.write_text("\n".join(rows), encoding="utf-8")
    return intervals_path, comparisons_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.draws < 1000:
        raise ValueError("Use at least 1,000 bootstrap draws.")
    paths = analyze(args.results_dir, args.draws, args.seed)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
