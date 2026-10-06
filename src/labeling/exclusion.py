"""Explicit, reproducible interval labeling for sliding EEG windows.

Labels: 0 interictal, 1 preictal, 2 ictal, -1 excluded. Times are seconds on
the patient clock when available, otherwise seconds from the EDF start. Preictal
means [onset - SPH, onset - onset_exclusion); configurable interictal and
postictal exclusion regions are left unlabeled. Boundary-overlapping windows
are excluded.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Seizure:
    onset: float
    end: float
    seizure_id: str
    recording_id: str | None = None


@dataclass(frozen=True)
class WindowLabel:
    label: int
    reason: str
    seizure_id: str | None = None
    seizure_onset: float | None = None
    seizure_end: float | None = None
    seizure_recording_id: str | None = None


def label_window(start: float, end: float, seizures: list[Seizure], horizon_minutes: int,
                 onset_exclusion_seconds: float = 0,
                 interictal_exclusion_seconds: float = 0,
                 postictal_exclusion_seconds: float = 0) -> WindowLabel:
    if end <= start:
        raise ValueError("window end must be after start")
    if horizon_minutes not in (5, 10, 15, 30):
        raise ValueError("horizon_minutes must be one of 5, 10, 15, 30")
    # Give fully annotated ictal intervals priority across all events.
    for seizure in seizures:
        if start >= seizure.onset and end <= seizure.end:
            return WindowLabel(2, "Fully within annotated ictal interval", seizure.seizure_id,
                               seizure.onset, seizure.end, seizure.recording_id)
    # Exclusions have precedence across events, preventing overlap from being relabeled.
    for seizure in seizures:
        ambiguous_start = seizure.onset - onset_exclusion_seconds
        if start < seizure.end and end > ambiguous_start:
            return WindowLabel(-1, "Window intersects onset exclusion or ictal boundary",
                               seizure.seizure_id, seizure.onset, seizure.end, seizure.recording_id)
        if start < seizure.end + postictal_exclusion_seconds and end > seizure.end:
            return WindowLabel(-1, "Window intersects configured postictal exclusion buffer",
                               seizure.seizure_id, seizure.onset, seizure.end, seizure.recording_id)
        pre_start = seizure.onset - horizon_minutes * 60
        if start < pre_start and end > pre_start - interictal_exclusion_seconds:
            return WindowLabel(-1, "Window intersects interictal-to-preictal exclusion buffer",
                               seizure.seizure_id, seizure.onset, seizure.end, seizure.recording_id)
    for seizure in seizures:
        pre_start = seizure.onset - horizon_minutes * 60
        pre_end = seizure.onset - onset_exclusion_seconds
        if start >= pre_start and end <= pre_end:
            return WindowLabel(1, "Within configured preictal horizon and outside onset exclusion",
                               seizure.seizure_id, seizure.onset, seizure.end, seizure.recording_id)
    return WindowLabel(0, "Outside annotated seizure, preictal, and exclusion intervals")
