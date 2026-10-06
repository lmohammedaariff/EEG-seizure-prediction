"""Optional post-hoc explanation helpers; explanations are not clinical localization."""
from __future__ import annotations
import torch
import numpy as np


def integrated_gradients(model, inputs, target="prediction_logits", steps=32):
    """Captum Integrated Gradients for all three inputs; non-active domains get zero attribution."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise RuntimeError("Install the explain extra: pip install -e '.[explain]'") from exc
    outputs = [torch.zeros_like(x) for x in inputs]
    names = ("temporal", "spectral", "time_frequency")
    for i, name in enumerate(names):
        if name not in model.active_domains:
            continue
        def forward(x):
            values = list(inputs)
            values[i] = x
            output = model(*values)[target]
            return output if output.ndim == 1 else output.squeeze(-1)
        outputs[i] = IntegratedGradients(forward).attribute(
            inputs[i], baselines=torch.zeros_like(inputs[i]), n_steps=steps)
    return tuple(outputs)


def summarize_importance(attributions):
    """Return channel, time, spectral-band and time-frequency importance arrays."""
    temporal, spectral, time_frequency = [a.detach().abs().mean(dim=0).cpu().numpy() for a in attributions]
    return {
        "temporal_channel_importance": temporal.mean(axis=-1),
        "temporal_time_importance": temporal.mean(axis=0),
        "spectral_channel_importance": spectral.mean(axis=-1),
        "spectral_band_importance": spectral.mean(axis=0),
        "time_frequency_channel_importance": time_frequency.mean(axis=(1, 2)),
        "time_frequency_region_importance": time_frequency.mean(axis=0),
        "time_frequency_frequency_importance": time_frequency.mean(axis=(0, 2)),
        "time_frequency_time_importance": time_frequency.mean(axis=(0, 1)),
    }


@torch.no_grad()
def attention_weights(model, temporal, spectral, time_frequency):
    model.eval()
    return model(temporal, spectral, time_frequency)["attention"]


def shap_values(model, inputs, background=None, max_samples=16):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install the explain extra: pip install -e '.[explain]'") from exc
    temporal, spectral, time_frequency = inputs
    indices = [i for i, name in enumerate(("temporal", "spectral", "time_frequency"))
               if name in model.active_domains]
    background = [((background or inputs)[i][:max_samples]) for i in indices]
    def score(xs):
        values = list(inputs[:])
        for i, x in zip(indices, xs):
            values[i] = x
        return torch.sigmoid(model(*values)["prediction_logits"])
    explainer = shap.GradientExplainer(score, background)
    values = explainer.shap_values([inputs[i][:max_samples] for i in indices])
    if isinstance(values, np.ndarray):
        values = [values]
    # SHAP versions differ in whether a scalar-output multi-input model returns a
    # flat list or a nested output list. Normalize to one tensor per active input.
    if values and isinstance(values[0], list):
        values = values[0]
    output = [torch.zeros_like(x[:max_samples]) for x in inputs]
    for i, value in zip(indices, values):
        output[i] = torch.as_tensor(value)
    return tuple(output)
