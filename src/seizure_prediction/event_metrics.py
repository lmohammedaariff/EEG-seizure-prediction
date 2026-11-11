from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import pandas as pd


def _with_alarm_persistence(
    records: Mapping[str, pd.DataFrame], persistence_windows: int
) -> dict[str, pd.DataFrame]:
    """Require consecutive valid positive windows before emitting an alarm.

    The probability at a window becomes the trailing minimum over the required
    sequence. A threshold-crossing alarm is therefore timestamped at the end
    of the last window needed to confirm persistence.
    """
    if persistence_windows < 1:
        raise ValueError("Alarm persistence must be at least one window.")
    if persistence_windows == 1:
        return dict(records)
    result: dict[str, pd.DataFrame] = {}
    for recording, frame in records.items():
        if not {"time_seconds", "label", "score"}.issubset(frame.columns):
            raise ValueError(f"Record {recording} is missing time_seconds, label, or score.")
        ordered = frame.sort_values("time_seconds", kind="stable").reset_index(drop=True)
        scores = ordered["score"].to_numpy(dtype=np.float64)
        labels = ordered["label"].to_numpy(dtype=np.int64)
        valid = (labels == 0) | (labels == 1)
        persistent_scores = np.full(scores.shape, -1.0, dtype=np.float64)
        for index in range(persistence_windows - 1, len(scores)):
            start = index - persistence_windows + 1
            if valid[start : index + 1].all():
                persistent_scores[index] = float(np.min(scores[start : index + 1]))
        ordered["score"] = persistent_scores
        result[recording] = ordered
    return result


def _merge_alarm_times(
    sorted_alarm_times: np.ndarray,
    window_seconds: float,
    merge_gap_seconds: float,
) -> list[tuple[float, float]]:
    if not len(sorted_alarm_times):
        return []
    breaks = np.flatnonzero(
        np.diff(sorted_alarm_times) > window_seconds + merge_gap_seconds
    )
    starts = sorted_alarm_times[np.r_[0, breaks + 1]]
    ends = sorted_alarm_times[np.r_[breaks, len(sorted_alarm_times) - 1]]
    return list(zip(starts.tolist(), (ends + window_seconds).tolist(), strict=True))


def _prepare_event_metrics(
    records: Mapping[str, pd.DataFrame],
    recording_offsets_seconds: Mapping[str, float],
    events: Sequence[tuple[float, float]],
    horizon_minutes: int,
    sph_minutes: int,
    window_seconds: float,
    alarm_merge_gap_minutes: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    list[tuple[float, float, float, float]],
    int,
]:
    if horizon_minutes <= 0 or sph_minutes < 0:
        raise ValueError("SOP must be positive and SPH cannot be negative.")
    if window_seconds <= 0 or alarm_merge_gap_minutes < 0:
        raise ValueError("Window length must be positive and alarm merge gap non-negative.")

    exposure_starts: list[np.ndarray] = []
    positive_window_starts: list[np.ndarray] = []
    alarm_time_parts: list[np.ndarray] = []
    alarm_score_parts: list[np.ndarray] = []
    for recording, frame in records.items():
        key = recording.lower()
        if key not in recording_offsets_seconds:
            raise ValueError(f"No recording offset found for {recording}.")
        offset = float(recording_offsets_seconds[key])
        required = {"time_seconds", "label", "score"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Record {recording} is missing {sorted(required - set(frame.columns))}.")
        starts = frame["time_seconds"].to_numpy(dtype=np.float64) + offset
        labels = frame["label"].to_numpy(dtype=np.int64)
        scores = frame["score"].to_numpy(dtype=np.float64)
        if not np.isfinite(starts).all() or not np.isfinite(scores).all():
            raise ValueError(f"Record {recording} contains non-finite times or scores.")

        exposure_starts.append(starts[labels == 0])
        positive_window_starts.append(starts[labels == 1])
        valid_alarm_scores = (labels == 0) | (labels == 1)
        alarm_time_parts.append(starts[valid_alarm_scores] + window_seconds)
        alarm_score_parts.append(scores[valid_alarm_scores])

    empty = np.empty(0, dtype=np.float64)
    exposure_times = (
        np.sort(np.concatenate(exposure_starts))
        if exposure_starts
        else empty
    )
    if len(exposure_times):
        exposure_ends = exposure_times + window_seconds
        furthest_ends = np.maximum.accumulate(exposure_ends)
        component_breaks = (
            np.flatnonzero(exposure_times[1:] > furthest_ends[:-1]) + 1
        )
        component_starts = exposure_times[np.r_[0, component_breaks]]
        component_end_indices = np.r_[component_breaks - 1, len(exposure_times) - 1]
        component_ends = furthest_ends[component_end_indices]
        exposure_seconds = float((component_ends - component_starts).sum())
    else:
        exposure_seconds = 0.0

    positive_starts = (
        np.sort(np.concatenate(positive_window_starts))
        if positive_window_starts
        else empty
    )
    sop_seconds = horizon_minutes * 60.0
    sph_seconds = sph_minutes * 60.0
    eligible_events: list[tuple[float, float, float, float]] = []
    for onset, offset in events:
        if not (math.isfinite(onset) and math.isfinite(offset) and offset > onset):
            raise ValueError("Seizure event onset/offset is invalid.")
        forecast_start = onset - sop_seconds - sph_seconds
        forecast_end = onset - sph_seconds
        first_candidate = int(np.searchsorted(positive_starts, forecast_start - 1e-6))
        has_labeled_forecast_time = (
            first_candidate < len(positive_starts)
            and positive_starts[first_candidate] + window_seconds
            <= forecast_end + 1e-6
        )
        if has_labeled_forecast_time:
            eligible_events.append((onset, offset, forecast_start, forecast_end))

    alarm_times = (
        np.concatenate(alarm_time_parts) if alarm_time_parts else empty
    )
    alarm_scores = (
        np.concatenate(alarm_score_parts) if alarm_score_parts else empty
    )
    order = np.argsort(alarm_times, kind="stable")
    return (
        alarm_times[order],
        alarm_scores[order],
        exposure_seconds / 3600.0,
        eligible_events,
        len(events),
    )


def _metrics_from_prepared(
    sorted_alarm_times: np.ndarray,
    exposure_hours: float,
    eligible_events: list[tuple[float, float, float, float]],
    annotated_event_count: int,
    window_seconds: float,
    alarm_merge_gap_minutes: float,
) -> dict[str, float | int]:
    alarms = _merge_alarm_times(
        sorted_alarm_times,
        window_seconds,
        alarm_merge_gap_minutes * 60.0,
    )
    detected_leads: list[float] = []
    matched_alarms: set[int] = set()
    for onset, _, forecast_start, forecast_end in eligible_events:
        matching = [
            (index, start)
            for index, (start, _) in enumerate(alarms)
            if index not in matched_alarms and forecast_start <= start < forecast_end
        ]
        if matching:
            alarm_index, alarm_start = min(matching, key=lambda item: item[1])
            matched_alarms.add(alarm_index)
            detected_leads.append((onset - alarm_start) / 60.0)

    false_alarms = len(alarms) - len(matched_alarms)
    detected_events = len(detected_leads)
    eligible_count = len(eligible_events)
    return {
        "event_sensitivity": (
            detected_events / eligible_count if eligible_count else float("nan")
        ),
        "detected_events": detected_events,
        "eligible_events": eligible_count,
        "all_annotated_events": annotated_event_count,
        "alarm_count": len(alarms),
        "false_alarms": false_alarms,
        "eligible_interictal_hours": exposure_hours,
        "false_alarms_per_hour": (
            false_alarms / exposure_hours if exposure_hours else float("nan")
        ),
        "mean_lead_time_minutes": (
            float(np.mean(detected_leads)) if detected_leads else float("nan")
        ),
        "median_lead_time_minutes": (
            float(np.median(detected_leads)) if detected_leads else float("nan")
        ),
    }


def event_forecast_metrics(
    records: Mapping[str, pd.DataFrame],
    recording_offsets_seconds: Mapping[str, float],
    events: Sequence[tuple[float, float]],
    horizon_minutes: int,
    sph_minutes: int,
    threshold: float,
    window_seconds: float,
    alarm_merge_gap_minutes: float = 1.0,
    alarm_persistence_windows: int = 1,
) -> dict[str, float | int]:
    """Score event forecasts using alarm onset, SOP/SPH, and interictal exposure.

    Record frames contain local `time_seconds`, horizon-specific integer `label`
    (0/1; -1, ictal, and ambiguous samples are excluded), and positive-class
    `score`. A score becomes available at the end of its EEG window.
    """
    if not 0 <= threshold <= 1 and not math.isclose(threshold, 1.000001):
        raise ValueError("Threshold must be a probability in [0, 1].")
    records = _with_alarm_persistence(records, alarm_persistence_windows)
    alarm_times, alarm_scores, exposure_hours, eligible_events, event_count = (
        _prepare_event_metrics(
            records,
            recording_offsets_seconds,
            events,
            horizon_minutes,
            sph_minutes,
            window_seconds,
            alarm_merge_gap_minutes,
        )
    )
    return _metrics_from_prepared(
        alarm_times[alarm_scores >= threshold],
        exposure_hours,
        eligible_events,
        event_count,
        window_seconds,
        alarm_merge_gap_minutes,
    )


def calibrate_thresholds(
    records: Mapping[str, pd.DataFrame],
    recording_offsets_seconds: Mapping[str, float],
    events: Sequence[tuple[float, float]],
    horizon_minutes: int,
    sph_minutes: int,
    false_alarm_budgets_per_hour: Sequence[float],
    window_seconds: float,
    alarm_merge_gap_minutes: float = 1.0,
    candidate_count: int = 81,
    alarm_persistence_windows: int = 1,
) -> dict[float, tuple[float, dict[str, float | int]]]:
    """Choose validation-only thresholds maximizing event sensitivity at FAR limits."""
    if candidate_count < 2:
        raise ValueError("At least two threshold candidates are required.")
    budgets = tuple(float(value) for value in false_alarm_budgets_per_hour)
    if not budgets or any(value < 0 or not math.isfinite(value) for value in budgets):
        raise ValueError("Provide finite non-negative false-alarm budgets.")
    records = _with_alarm_persistence(records, alarm_persistence_windows)
    alarm_times, alarm_scores, exposure_hours, eligible_events, event_count = (
        _prepare_event_metrics(
            records,
            recording_offsets_seconds,
            events,
            horizon_minutes,
            sph_minutes,
            window_seconds,
            alarm_merge_gap_minutes,
        )
    )
    grid = np.linspace(0.0, 1.0, candidate_count).tolist()
    grid.append(1.000001)
    evaluated = [
        (
            threshold,
            _metrics_from_prepared(
                alarm_times[alarm_scores >= threshold],
                exposure_hours,
                eligible_events,
                event_count,
                window_seconds,
                alarm_merge_gap_minutes,
            ),
        )
        for threshold in grid
    ]
    selected: dict[float, tuple[float, dict[str, float | int]]] = {}
    for budget in budgets:
        feasible = [
            item
            for item in evaluated
            if math.isfinite(float(item[1]["false_alarms_per_hour"]))
            and float(item[1]["false_alarms_per_hour"]) <= budget + 1e-12
        ]
        if not feasible:
            raise ValueError(
                f"Validation exposure cannot estimate a false-alarm rate for "
                f"{budget:g} alarms/hour."
            )
        threshold, metrics = max(
            feasible,
            key=lambda item: (
                (
                    float(item[1]["event_sensitivity"])
                    if math.isfinite(float(item[1]["event_sensitivity"]))
                    else -1.0
                ),
                -float(item[1]["false_alarms_per_hour"]),
                item[0],
            ),
        )
        selected[budget] = (threshold, metrics)
    return selected
