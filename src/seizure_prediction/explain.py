from __future__ import annotations

import numpy as np
from scipy.signal import stft
import torch
from torch import nn


def integrated_gradients(
    model: nn.Module,
    signal: torch.Tensor,
    target_class: int | None = None,
    steps: int = 24,
    head: str = "state",
    horizon_index: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Approximate Integrated Gradients for one [1, channels, samples] window."""
    if signal.ndim != 3 or signal.shape[0] != 1:
        raise ValueError("Integrated Gradients expects one EEG window with batch size 1.")
    if steps < 2:
        raise ValueError("Integrated Gradients requires at least two integration steps.")
    if head not in {"state", "prediction", "horizon"}:
        raise ValueError("Integrated Gradients head must be state, prediction, or horizon.")
    if head == "horizon" and horizon_index is None:
        raise ValueError("A horizon index is required for multi-horizon attribution.")

    def selected_logits(output: dict[str, object]) -> torch.Tensor:
        key = {
            "state": "state_logits",
            "prediction": "prediction_logits",
            "horizon": "horizon_logits",
        }[head]
        logits = output.get(key)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Model output did not contain {key}.")
        if head == "horizon":
            if logits.ndim != 3 or horizon_index is None or not 0 <= horizon_index < logits.shape[1]:
                raise ValueError("Requested multi-horizon output index is not available.")
            return logits[:, horizon_index, :]
        return logits

    was_training = model.training
    model.eval()
    baseline = torch.zeros_like(signal)
    delta = signal - baseline
    total_gradient = torch.zeros_like(signal)
    predicted_class = 0
    try:
        with torch.no_grad():
            logits = selected_logits(model(signal))
            predicted_class = int(logits.argmax(dim=1).item())
        selected_class = predicted_class if target_class is None else target_class
        for alpha in torch.linspace(0.0, 1.0, steps, device=signal.device):
            point = (baseline + alpha * delta).detach().requires_grad_(True)
            logits = selected_logits(model(point))
            score = logits[:, selected_class].sum()
            gradient = torch.autograd.grad(score, point)[0]
            total_gradient += gradient
        attribution = delta * total_gradient / steps
        return attribution.detach(), predicted_class
    finally:
        model.train(was_training)


def channel_importance(attribution: torch.Tensor) -> torch.Tensor:
    """Aggregate absolute input attribution per channel."""
    if attribution.ndim != 3:
        raise ValueError("Expected attribution [batch, channels, samples].")
    return attribution.abs().mean(dim=(0, 2))


def time_frequency_importance(
    attribution: torch.Tensor, sampling_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a mean channel time-frequency map of absolute input attribution."""
    if attribution.ndim != 3 or attribution.shape[0] != 1:
        raise ValueError("Expected one attribution window [1, channels, samples].")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    saliency = attribution.detach().abs().squeeze(0).cpu().numpy()
    frequencies, times, transformed = stft(
        saliency,
        fs=sampling_rate,
        nperseg=min(128, saliency.shape[-1]),
        noverlap=min(96, max(0, min(128, saliency.shape[-1]) - 1)),
        axis=-1,
    )
    importance = np.abs(transformed).mean(axis=0)
    return frequencies, times, importance


class _TargetOutput(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        target_class: int,
        head: str,
        horizon_index: int | None = None,
    ) -> None:
        super().__init__()
        if head not in {"state", "prediction", "horizon"}:
            raise ValueError("SHAP head must be 'state', 'prediction', or 'horizon'.")
        if head == "horizon" and horizon_index is None:
            raise ValueError("A horizon index is required for multi-horizon SHAP.")
        self.model = model
        self.target_class = target_class
        self.head = head
        self.horizon_index = horizon_index

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        output = self.model(signal)
        key = {
            "state": "state_logits",
            "prediction": "prediction_logits",
            "horizon": "horizon_logits",
        }[self.head]
        logits = output.get(key)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"The model did not return {key}.")
        if self.head == "horizon":
            if self.horizon_index is None or not 0 <= self.horizon_index < logits.shape[1]:
                raise ValueError("Requested multi-horizon output index is not available.")
            logits = logits[:, self.horizon_index, :]
        return logits[:, self.target_class : self.target_class + 1]


def shap_attributions(
    model: nn.Module,
    signal: torch.Tensor,
    target_class: int,
    head: str = "state",
    nsamples: int = 32,
    horizon_index: int | None = None,
) -> torch.Tensor:
    """Compute optional GradientSHAP attributions against a zero baseline."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP support is optional. Install it with `pip install -e \".[explain]\"`."
        ) from exc
    if signal.ndim != 3 or signal.shape[0] != 1:
        raise ValueError("SHAP expects one EEG window with batch size 1.")
    if nsamples < 1:
        raise ValueError("SHAP sample count must be positive.")
    wrapper = _TargetOutput(model, target_class, head, horizon_index).eval()
    background = torch.zeros_like(signal)
    explainer = shap.GradientExplainer(wrapper, background)
    values = explainer.shap_values(signal, nsamples=nsamples)
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Unexpected SHAP output count for the selected class.")
        values = values[0]
    array = np.asarray(values)
    if array.shape == (*signal.shape, 1):
        array = array[..., 0]
    if array.shape != tuple(signal.shape):
        raise ValueError(f"Unexpected SHAP attribution shape: {array.shape}.")
    return torch.as_tensor(array, dtype=signal.dtype, device=signal.device)
