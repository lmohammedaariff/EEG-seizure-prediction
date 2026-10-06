from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    horizons = cfg["data"]["preictal_horizons_minutes"]
    if not horizons or any(int(h) not in (5, 10, 15, 30) for h in horizons):
        raise ValueError("preictal_horizons_minutes must be selected from 5, 10, 15, 30")
    data = cfg["data"]
    if any(data[k] < 0 for k in ("onset_exclusion_seconds", "interictal_exclusion_seconds",
                                  "postictal_exclusion_seconds")):
        raise ValueError("exclusion buffers must be non-negative seconds")
    if data["onset_exclusion_seconds"] >= min(horizons) * 60:
        raise ValueError("onset exclusion must be shorter than the smallest configured horizon")
    if any(int(v) < 0 for v in data.get("buffer_sensitivity_seconds", [])):
        raise ValueError("buffer_sensitivity_seconds must contain non-negative seconds")
    if data["window_seconds"] <= 0 or data["sampling_rate"] <= 0:
        raise ValueError("window_seconds and sampling_rate must be positive")
    if not 1 <= data["channels"] <= 18:
        raise ValueError("channels must be between 1 and the 18-channel standardized montage")
    if not 0 < data["low_hz"] < data["high_hz"] < data["sampling_rate"] / 2:
        raise ValueError("filter cutoffs must satisfy 0 < low_hz < high_hz < Nyquist")
    if cfg["training"]["epochs"] < 1 or cfg["training"]["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 <= cfg["data"]["overlap"] < 1:
        raise ValueError("overlap must be in [0, 1)")
    return cfg
