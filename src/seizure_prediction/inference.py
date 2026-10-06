from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from scipy.signal import stft
from .data import preprocess, normalize_window, standardized_montage, CHB_CHANNELS
from .model import MultiDomainNet


def transform_edf(path: str | Path, cfg: dict, strict_channels: bool = False):
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    x, missing_channels = standardized_montage(raw, cfg["data"]["channels"], strict_channels)
    d = cfg["data"]
    fs = int(d["sampling_rate"])
    x = preprocess(x, float(raw.info["sfreq"]), fs, d["low_hz"], d["high_hz"], d["notch_hz"])
    n = int(d["window_seconds"] * fs)
    step = max(1, int(n * (1 - d["overlap"])))
    f = np.fft.rfftfreq(n, 1 / fs)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    temporal, spectral, tf, times = [], [], [], []
    for start_idx in range(0, x.shape[1] - n + 1, step):
        w = normalize_window(x[:, start_idx:start_idx + n])
        power = np.abs(np.fft.rfft(w * np.hanning(n)[None, :], axis=-1)) ** 2 / n
        spectral.append(np.stack([np.log1p(power[:, (f >= lo) & (f < hi)].mean(-1)) for lo, hi in bands], -1))
        nperseg = min(64, n)
        _, _, z = stft(w, fs=fs, nperseg=nperseg, noverlap=min(48, n - 1), axis=-1)
        keep_bins = min(z.shape[1], int(d["high_hz"] / (fs / nperseg)) + 1)
        image = np.log1p(np.abs(z))[:, :keep_bins, :]
        if image.shape[-1] < 33:
            image = np.pad(image, ((0, 0), (0, 0), (0, 33 - image.shape[-1])))
        temporal.append(w.astype(np.float16)); tf.append(image.astype(np.float16))
        times.append((start_idx / fs, (start_idx + n) / fs))
    return (np.stack(temporal), np.asarray(spectral, np.float32), np.stack(tf),
            pd.DataFrame(times, columns=["start_time", "end_time"]), missing_channels)


def model_from_checkpoint(saved, device="cpu"):
    cfg, stage = saved["config"], saved["stage"]
    domains = {"temporal_only": ("temporal",), "temporal_spectral": ("temporal", "spectral"),
               "cnn": ("temporal",), "lstm": ("temporal",), "cnn_lstm": ("temporal",)}.get(
                   stage, ("temporal", "spectral", "time_frequency"))
    arch = stage if stage in ("cnn", "lstm", "cnn_lstm") else "cnn"
    n_subjects = saved["model"]["subject_head.2.weight"].shape[0]
    model = MultiDomainNet(channels=saved["channels"], width=cfg["model"]["width"],
                           dropout=cfg["model"]["dropout"], n_subjects=n_subjects,
                           fusion="attention" if stage in ("multi_domain_attention", "complete") else "mean",
                           active_domains=domains, temporal_architecture=arch,
                           domain_adversarial=stage == "complete" and cfg["model"]["domain_adversarial"])
    model.load_state_dict(saved["model"]); model.to(device).eval()
    return model, domains


def predict_edf(path: str | Path, checkpoint: str | Path, device="cpu",
                strict_channels: bool = False) -> pd.DataFrame:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    arrays = transform_edf(path, cfg, strict_channels)
    model, domains = model_from_checkpoint(saved, device)
    pred, states, attention = [], [], []
    with torch.no_grad():
        for start in range(0, len(arrays[0]), cfg["training"]["batch_size"]):
            batch = [torch.as_tensor(a[start:start + cfg["training"]["batch_size"]], device=device).float()
                     for a in arrays[:3]]
            result = model(*batch)
            pred.extend(torch.sigmoid(result["prediction_logits"]).cpu().numpy())
            states.extend(result["state_logits"].argmax(1).cpu().numpy())
            attention.extend(result["attention"].cpu().numpy())
    frame = arrays[3]
    frame["recording_id"] = Path(path).name
    frame["missing_montage_channels"] = ";".join(arrays[4])
    frame["preictal_probability"] = pred
    frame["prediction"] = np.where(np.asarray(pred) >= cfg["training"]["threshold"], "preictal", "non-preictal")
    frame["three_state_prediction"] = np.asarray(["interictal", "preictal", "ictal"])[states]
    for i, name in enumerate(domains):
        frame[f"attention_{name}"] = np.asarray(attention)[:, i]
    frame["horizon_minutes"] = saved["horizon"]
    frame["held_out_patient"] = saved["patient"]
    return frame


def explain_edf_window(path: str | Path, checkpoint: str | Path, window_index: int,
                       device="cpu", use_shap: bool = False):
    from .explain import integrated_gradients, summarize_importance, shap_values
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    arrays = transform_edf(path, cfg)
    if not 0 <= window_index < len(arrays[0]):
        raise IndexError(f"window_index must be in [0, {len(arrays[0]) - 1}]")
    model, _ = model_from_checkpoint(saved, device)
    one = tuple(torch.as_tensor(a[window_index:window_index + 1], device=device).float() for a in arrays[:3])
    attribution = integrated_gradients(model, one)
    result = {"integrated_gradients": summarize_importance(attribution), "shap": None}
    if use_shap:
        count = min(16, len(arrays[0]))
        inds = np.linspace(0, len(arrays[0]) - 1, count).round().astype(int)
        background = tuple(torch.as_tensor(a[inds], device=device).float() for a in arrays[:3])
        shap_attrs = shap_values(model, one, background=background)
        result["shap"] = summarize_importance(shap_attrs)
    return result
