from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


CHANNELS = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
)

STATE_NAMES = ("Interictal", "Preictal", "Ictal")
PREDICTION_NAMES = ("Interictal", "Preictal")
SINGLE_HORIZON_VARIANTS = (
    "temporal_only",
    "temporal_spectral",
    "multi_domain",
    "subject_invariant",
    "cnn",
    "lstm",
    "cnn_lstm",
)
MULTIHORIZON_MODEL_VARIANTS = (
    "multi_horizon_temporal",
    "multi_horizon_temporal_spectral",
    "multi_horizon_multi_domain",
    "multi_horizon",
    "multi_horizon_cnn",
    "multi_horizon_lstm",
    "multi_horizon_cnn_lstm",
)
MULTIHORIZON_ABLATION_VARIANTS = MULTIHORIZON_MODEL_VARIANTS[:4]
MULTIHORIZON_BASELINE_VARIANTS = MULTIHORIZON_MODEL_VARIANTS[4:]
MODEL_VARIANTS = (*SINGLE_HORIZON_VARIANTS, *MULTIHORIZON_MODEL_VARIANTS)


@dataclass(frozen=True)
class ExperimentConfig:
    sampling_rate: int = 128
    window_seconds: float = 10.0
    overlap: float = 0.5
    notch_hz: float = 60.0
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 45.0
    preictal_minutes: int = 30
    sph_minutes: int = 5
    interictal_buffer_minutes: int = 60
    horizons_minutes: tuple[int, ...] = (5, 10, 15, 30)
    min_ictal_overlap_seconds: float = 1.0
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    subject_loss_weight: float = 0.2
    alarm_merge_gap_minutes: float = 1.0
    far_budgets_per_hour: tuple[float, ...] = (0.1, 0.5, 1.0)
    random_seed: int = 42
    model_variant: str = "subject_invariant"

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0 or self.window_seconds <= 0:
            raise ValueError("Sampling rate and window length must be positive.")
        if round(self.sampling_rate * self.window_seconds) < 16:
            raise ValueError("Window length must include at least 16 samples.")
        if not 0 <= self.overlap < 1:
            raise ValueError("Window overlap must be in [0, 1).")
        if round(self.sampling_rate * self.window_seconds * (1 - self.overlap)) < 1:
            raise ValueError("Window step must be at least one sample.")
        if self.bandpass_low_hz <= 0 or self.bandpass_high_hz <= self.bandpass_low_hz:
            raise ValueError("Band-pass filter frequencies are invalid.")
        if self.notch_hz <= 0:
            raise ValueError("Notch frequency must be positive.")
        if self.preictal_minutes <= 0 or self.sph_minutes < 0:
            raise ValueError("SOP must be positive and SPH cannot be negative.")
        if self.interictal_buffer_minutes < 0:
            raise ValueError("Interictal exclusion buffer cannot be negative.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("Batch size and epoch count must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rate must be positive and weight decay non-negative.")
        if self.alarm_merge_gap_minutes < 0:
            raise ValueError("Alarm merge gap cannot be negative.")
        if (
            not self.far_budgets_per_hour
            or any(not isinstance(budget, (int, float)) or budget < 0 for budget in self.far_budgets_per_hour)
            or len(set(self.far_budgets_per_hour)) != len(self.far_budgets_per_hour)
        ):
            raise ValueError("False-alarm budgets must be unique, non-negative values.")
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model variant: {self.model_variant}")
        if (
            not self.horizons_minutes
            or any(not isinstance(h, int) or h <= 0 for h in self.horizons_minutes)
            or len(set(self.horizons_minutes)) != len(self.horizons_minutes)
        ):
            raise ValueError("At least one positive preictal horizon is required.")

    @property
    def window_samples(self) -> int:
        return round(self.sampling_rate * self.window_seconds)

    @property
    def step_samples(self) -> int:
        return round(self.window_samples * (1 - self.overlap))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["horizons_minutes"] = list(self.horizons_minutes)
        return result

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> ExperimentConfig:
        normalized = dict(values)
        if "horizons_minutes" in normalized:
            normalized["horizons_minutes"] = tuple(normalized["horizons_minutes"])
        if "far_budgets_per_hour" in normalized:
            normalized["far_budgets_per_hour"] = tuple(normalized["far_budgets_per_hour"])
        return cls(**normalized)


@dataclass(frozen=True)
class Paths:
    data_dir: Path = field(default_factory=lambda: Path("data/chbmit"))
    cache_dir: Path = field(default_factory=lambda: Path("artifacts/cache"))
    output_dir: Path = field(default_factory=lambda: Path("artifacts/results"))


ModelVariant = Literal[
    "temporal_only",
    "temporal_spectral",
    "multi_domain",
    "subject_invariant",
    "multi_horizon_temporal",
    "multi_horizon_temporal_spectral",
    "multi_horizon_multi_domain",
    "multi_horizon",
    "multi_horizon_cnn",
    "multi_horizon_lstm",
    "multi_horizon_cnn_lstm",
    "cnn",
    "lstm",
    "cnn_lstm",
]
