from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

INTERICTAL = 0
PREICTAL = 1
ICTAL = 2
EXCLUDED = -1


@dataclass(frozen=True)
class Seizure:
    onset_seconds: float
    offset_seconds: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.onset_seconds)
            or not math.isfinite(self.offset_seconds)
            or self.onset_seconds < 0
            or self.offset_seconds <= self.onset_seconds
        ):
            raise ValueError("A seizure must have a non-negative onset before its offset.")


def label_windows(
    starts_seconds: np.ndarray,
    window_seconds: float,
    seizures: list[Seizure],
    preictal_minutes: int,
    sph_minutes: int,
    interictal_buffer_minutes: int,
    min_ictal_overlap_seconds: float = 1.0,
) -> np.ndarray:
    """Return 0=interictal, 1=preictal, 2=ictal, -1=excluded/ambiguous."""
    starts = np.asarray(starts_seconds, dtype=np.float64)
    if not np.isfinite(starts).all():
        raise ValueError("Window start times must be finite.")
    if window_seconds <= 0:
        raise ValueError("Window length must be positive.")
    if preictal_minutes <= 0 or sph_minutes < 0 or interictal_buffer_minutes < 0:
        raise ValueError("Invalid SOP, SPH, or exclusion buffer.")
    if not 0 < min_ictal_overlap_seconds <= window_seconds:
        raise ValueError("Minimum ictal overlap must be within the EEG window.")

    ends = starts + window_seconds
    labels = np.full(starts.shape, EXCLUDED, dtype=np.int8)
    ictal_overlap = np.zeros(starts.shape, dtype=np.float64)

    for seizure in seizures:
        overlap = np.maximum(
            0.0,
            np.minimum(ends, seizure.offset_seconds)
            - np.maximum(starts, seizure.onset_seconds),
        )
        ictal_overlap = np.maximum(ictal_overlap, overlap)

    labels[ictal_overlap >= min_ictal_overlap_seconds] = ICTAL
    eligible = labels == EXCLUDED
    for seizure in seizures:
        preictal_start = seizure.onset_seconds - (
            preictal_minutes + sph_minutes
        ) * 60
        preictal_end = seizure.onset_seconds - sph_minutes * 60
        wholly_preictal = (starts >= preictal_start) & (ends <= preictal_end)
        labels[eligible & wholly_preictal] = PREICTAL

    eligible = labels == EXCLUDED
    safe_interictal = np.ones(starts.shape, dtype=bool)
    buffer_seconds = interictal_buffer_minutes * 60
    for seizure in seizures:
        preictal_start = seizure.onset_seconds - (
            preictal_minutes + sph_minutes
        ) * 60
        safe_interictal &= (
            (ends <= preictal_start - buffer_seconds)
            | (starts >= seizure.offset_seconds + buffer_seconds)
        )
    labels[eligible & safe_interictal] = INTERICTAL
    return labels


def prediction_labels(state_labels: np.ndarray) -> np.ndarray:
    """Map state labels to a binary preictal task; discard ictal/ambiguous."""
    states = np.asarray(state_labels, dtype=np.int64)
    keep = (states == INTERICTAL) | (states == PREICTAL)
    return np.where(states[keep] == PREICTAL, 1, 0).astype(np.int64)
