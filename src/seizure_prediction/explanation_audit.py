from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from .config import ExperimentConfig
from .datasets import MultiHorizonEEGDataset
from .explain import channel_importance, integrated_gradients


def audit_event_explanations(
    model: nn.Module,
    dataset: MultiHorizonEEGDataset,
    horizon_index: int,
    config: ExperimentConfig,
    max_windows_per_event: int = 3,
    top_channels: int = 3,
    random_masks: int = 20,
    integrated_gradient_steps: int = 16,
) -> list[dict[str, object]]:
    """Measure within-event channel-rank stability and channel-mask faithfulness."""
    if not 0 <= horizon_index < len(config.horizons_minutes):
        raise ValueError("Explanation horizon index is outside the configured horizons.")
    if max_windows_per_event < 2:
        raise ValueError("At least two windows per event are required for stability.")
    if top_channels < 1 or random_masks < 1:
        raise ValueError("Top channel and random-mask counts must be positive.")

    rows: list[dict[str, object]] = []
    horizon = config.horizons_minutes[horizon_index]
    for record_index, recording in enumerate(dataset.recording_names):
        key = recording.lower()
        recording_offset = dataset.recording_offsets_seconds[key]
        record_mask = dataset.recording_indices == record_index
        local_times = dataset.starts_seconds
        labels = dataset.horizon_labels[:, horizon_index]
        for event_index, (onset_local, _) in enumerate(
            dataset.seizures_by_recording.get(key, [])
        ):
            onset = recording_offset + onset_local
            forecast_start = onset - (
                horizon + config.sph_minutes
            ) * 60.0
            forecast_end = onset - config.sph_minutes * 60.0
            candidate_indices = np.flatnonzero(
                record_mask
                & (labels == 1)
                & (local_times + recording_offset >= forecast_start)
                & (
                    local_times
                    + recording_offset
                    + config.window_seconds
                    <= forecast_end
                )
            )
            if len(candidate_indices) < 2:
                continue
            selected = _spaced_indices(
                candidate_indices,
                local_times,
                max_windows_per_event,
                config.window_seconds,
            )
            importances = []
            faithfulness_drops = []
            random_drops = []
            top_channel_ids = []
            seed = (
                sum(recording.encode("utf-8"))
                + event_index * 7919
                + horizon * 101
            )
            rng = np.random.default_rng(seed)
            for sample_index in selected:
                signal = dataset.window(int(sample_index))
                tensor = torch.from_numpy(signal[None]).to(next(model.parameters()).device)
                attribution, _ = integrated_gradients(
                    model,
                    tensor,
                    target_class=1,
                    steps=integrated_gradient_steps,
                    head="horizon",
                    horizon_index=horizon_index,
                )
                importance = channel_importance(attribution).cpu().numpy()
                importances.append(importance)
                chosen = np.argsort(importance)[-top_channels:]
                top_channel_ids.append(set(int(index) for index in chosen))
                baseline_score = _horizon_probability(model, tensor, horizon_index)
                masked = tensor.clone()
                masked[:, chosen, :] = 0
                faithfulness_drops.append(
                    baseline_score - _horizon_probability(model, masked, horizon_index)
                )
                random_event_drops = []
                n_channels = tensor.shape[1]
                for _ in range(random_masks):
                    random_set = rng.choice(
                        n_channels,
                        size=min(top_channels, n_channels),
                        replace=False,
                    )
                    random_masked = tensor.clone()
                    random_masked[:, random_set, :] = 0
                    random_event_drops.append(
                        baseline_score
                        - _horizon_probability(model, random_masked, horizon_index)
                    )
                random_drops.append(float(np.mean(random_event_drops)))

            correlations = []
            jaccards = []
            for first in range(len(importances)):
                for second in range(first + 1, len(importances)):
                    correlation = spearmanr(
                        importances[first], importances[second]
                    ).statistic
                    if np.isfinite(correlation):
                        correlations.append(float(correlation))
                    union = top_channel_ids[first] | top_channel_ids[second]
                    jaccards.append(
                        len(top_channel_ids[first] & top_channel_ids[second]) / len(union)
                        if union
                        else 1.0
                    )
            rows.append(
                {
                    "patient_id": dataset.patient_id,
                    "recording": recording,
                    "event_index_in_recording": event_index + 1,
                    "horizon_minutes": horizon,
                    "windows_audited": len(selected),
                    "channel_rank_spearman_mean": (
                        float(np.mean(correlations)) if correlations else float("nan")
                    ),
                    "top_channel_jaccard_mean": (
                        float(np.mean(jaccards)) if jaccards else float("nan")
                    ),
                    "top_channel_mask_risk_drop_mean": float(np.mean(faithfulness_drops)),
                    "random_channel_mask_risk_drop_mean": float(np.mean(random_drops)),
                    "top_channel_mask_advantage": float(
                        np.mean(faithfulness_drops) - np.mean(random_drops)
                    ),
                    "top_channels_by_window": ";".join(
                        ",".join(str(index) for index in sorted(channels))
                        for channels in top_channel_ids
                    ),
                }
            )
    return rows


def _spaced_indices(
    candidates: np.ndarray,
    starts_seconds: np.ndarray,
    limit: int,
    minimum_gap_seconds: float,
) -> list[int]:
    chosen: list[int] = []
    for index in candidates:
        if not chosen or starts_seconds[index] - starts_seconds[chosen[-1]] >= minimum_gap_seconds:
            chosen.append(int(index))
    if len(chosen) > limit:
        positions = np.linspace(0, len(chosen) - 1, num=limit, dtype=np.int64)
        chosen = [chosen[int(position)] for position in positions]
    return chosen


def _horizon_probability(
    model: nn.Module,
    signal: torch.Tensor,
    horizon_index: int,
) -> float:
    with torch.inference_mode():
        logits = model(signal)["horizon_logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Model does not expose multi-horizon prediction logits.")
        return float(torch.softmax(logits[:, horizon_index], dim=1)[0, 1].item())
