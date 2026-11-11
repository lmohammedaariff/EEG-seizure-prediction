from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import ExperimentConfig, STATE_NAMES
from .edf import preprocess_edf, window_recording
from .training import load_checkpoint


def predict_signal(
    signal: np.ndarray,
    config: ExperimentConfig,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 64,
    selected_horizon_minutes: int | None = None,
    score_threshold: float = 0.5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, list[str]]:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")
    windows, _, starts = window_recording(signal, [], config)
    state_probabilities: list[np.ndarray] = []
    prediction_probabilities: list[np.ndarray] = []
    horizon_probabilities: list[np.ndarray] = []
    domain_attention: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                np.asarray(windows[offset : offset + batch_size], dtype=np.float32)
            ).to(device)
            output = model(batch)
            state_logits = output["state_logits"]
            prediction_logits = output["prediction_logits"]
            if not isinstance(state_logits, torch.Tensor) or not isinstance(
                prediction_logits, torch.Tensor
            ):
                raise TypeError("Checkpoint model did not return both prediction heads.")
            state_probabilities.append(torch.softmax(state_logits, dim=1).cpu().numpy())
            prediction_probabilities.append(
                torch.softmax(prediction_logits, dim=1).cpu().numpy()
            )
            horizon_logits = output.get("horizon_logits")
            if isinstance(horizon_logits, torch.Tensor):
                horizon_probabilities.append(
                    torch.softmax(horizon_logits, dim=2)[:, :, 1].cpu().numpy()
                )
            attention = output["attention"]
            if isinstance(attention, torch.Tensor):
                domain_attention.append(attention.cpu().numpy())

    state_probs = np.concatenate(state_probabilities)
    prediction_probs = np.concatenate(prediction_probabilities)
    horizon_probs = np.concatenate(horizon_probabilities) if horizon_probabilities else None
    if horizon_probs is not None:
        if horizon_probs.shape[1] != len(config.horizons_minutes):
            raise ValueError("Checkpoint horizon heads do not match its saved configuration.")
        requested_horizon = (
            selected_horizon_minutes
            if selected_horizon_minutes is not None
            else config.preictal_minutes
        )
        selected_horizon = (
            config.horizons_minutes.index(requested_horizon)
            if requested_horizon in config.horizons_minutes
            else len(config.horizons_minutes) - 1
        )
        selected_scores = horizon_probs[:, selected_horizon]
        prediction_probs = np.column_stack((1.0 - selected_scores, selected_scores))
    frame = pd.DataFrame(
        {
            "time_minutes": (starts + config.window_seconds / 2) / 60,
            "predicted_state": [STATE_NAMES[index] for index in state_probs.argmax(axis=1)],
            "state_interictal_probability": state_probs[:, 0],
            "state_preictal_probability": state_probs[:, 1],
            "state_ictal_probability": state_probs[:, 2],
            "prediction": np.where(
                prediction_probs[:, 1] >= score_threshold,
                "Preictal",
                "Interictal",
            ),
            "preictal_probability": prediction_probs[:, 1],
            "interictal_probability": prediction_probs[:, 0],
        }
    )
    if horizon_probs is not None:
        for index, horizon in enumerate(config.horizons_minutes):
            frame[f"preictal_probability_sop_{horizon}m"] = horizon_probs[:, index]
    attention_array = np.concatenate(domain_attention) if domain_attention else None
    domain_names = (
        ["temporal", "spectral", "time_frequency"][: attention_array.shape[1]]
        if attention_array is not None
        else []
    )
    if not hasattr(model, "variant"):
        domain_names = []
    return frame, windows, attention_array, domain_names


def predict_edf(
    edf_path: Path,
    checkpoint_path: Path,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, list[str], list[str]]:
    model, payload, device = load_checkpoint(checkpoint_path)
    config = ExperimentConfig.from_dict(payload["config"])
    signal, matched_channels = preprocess_edf(edf_path, config)
    frame, windows, attention, domains = predict_signal(
        signal, config, model, device, batch_size=batch_size
    )
    return frame, windows, attention, domains, matched_channels


def integrated_gradient_importance(
    model: nn.Module,
    window: np.ndarray,
    device: torch.device,
    steps: int = 24,
) -> tuple[np.ndarray, int]:
    from .explain import channel_importance, integrated_gradients

    signal = torch.from_numpy(np.asarray(window[None], dtype=np.float32)).to(device)
    attribution, predicted_class = integrated_gradients(model, signal, steps=steps)
    importance = channel_importance(attribution).cpu().numpy()
    return importance, predicted_class
