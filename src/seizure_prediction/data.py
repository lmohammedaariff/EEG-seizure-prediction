"""CHB-MIT summary parsing, EDF preprocessing, windows and auditable metadata."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, iirnotch, lfilter, stft
from labeling.exclusion import Seizure, label_window


def parse_summary(path: Path) -> dict[str, list[Seizure]]:
    text = path.read_text(errors="replace")
    result: dict[str, list[Seizure]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        file_match = re.match(r"File Name:\s*(.+\.edf)", line, re.I)
        if file_match:
            current = file_match.group(1).strip()
            result[current] = []
            continue
        if current is None:
            continue
        start = re.match(r"Seizure(?:\s*(\d+))?\s*Start Time:\s*(\d+)\s*seconds", line, re.I)
        finish = re.match(r"Seizure(?:\s*(\d+))?\s*End Time:\s*(\d+)\s*seconds", line, re.I)
        if start:
            idx, sec = start.groups()
            idx = idx or str(len(result[current]) + 1)
            result[current].append(Seizure(float(sec), -1.0, f"{current}:sz{idx}", current))
        elif finish:
            idx, sec = finish.groups()
            if idx is None:
                pending = next((j for j in range(len(result[current]) - 1, -1, -1)
                                if result[current][j].end < 0), None)
                idx = result[current][pending].seizure_id.rsplit("sz", 1)[-1] if pending is not None else None
            if idx is None:
                raise ValueError(f"Seizure end without a matching start in {path}: {line}")
            for j in range(len(result[current]) - 1, -1, -1):
                if result[current][j].seizure_id == f"{current}:sz{idx}":
                    old = result[current][j]
                    result[current][j] = Seizure(old.onset, float(sec), old.seizure_id, current)
                    break
    return result


def parse_summary_timeline(path: Path):
    """Return recording-relative events and patient-clock offsets where summaries provide them."""
    local_events = parse_summary(path)
    order, clocks = [], {}
    current = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        file_match = re.match(r"File Name:\s*(.+\.edf)", line, re.I)
        if file_match:
            current = file_match.group(1).strip()
            order.append(current)
            continue
        start_match = re.match(r"File Start Time:\s*(\d{1,2}:\d{2}:\d{2})", line, re.I)
        if current and start_match:
            h, m, s = (int(part) for part in start_match.group(1).split(":"))
            clocks[current] = h * 3600 + m * 60 + s
    if not order or not all(name in clocks for name in order):
        return local_events, {}, []
    offsets, day_offset, last_clock = {}, 0.0, None
    for name in order:
        clock = clocks[name]
        if last_clock is not None and clock < last_clock - 12 * 3600:
            day_offset += 86400
        offsets[name] = day_offset + clock
        last_clock = clock
    global_events = [Seizure(offsets[name] + event.onset, offsets[name] + event.end,
                             event.seizure_id, name)
                     for name in order for event in local_events.get(name, [])]
    return local_events, offsets, global_events


def preprocess(data: np.ndarray, fs: float, target_fs: int, low: float, high: float,
               notch: float) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if fs != target_fs:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(round(fs)), int(target_fs))
        x = resample_poly(x, target_fs // g, int(round(fs)) // g, axis=-1).astype(np.float32)
    nyq = target_fs / 2
    sos = butter(4, [low / nyq, min(high / nyq, 0.99)], btype="bandpass", output="sos")
    # Causal filtering avoids using future samples when scoring the current window.
    x = sosfilt(sos, x, axis=-1).astype(np.float32)
    if 0 < notch < nyq:
        b, a = iirnotch(notch, 30, fs=target_fs)
        x = lfilter(b, a, x, axis=-1).astype(np.float32)
    return x


def normalize_window(x: np.ndarray) -> np.ndarray:
    """Robustly scale one input window, without statistics from future recording data."""
    med = np.median(x, axis=-1, keepdims=True)
    scale = np.median(np.abs(x - med), axis=-1, keepdims=True) * 1.4826
    return np.clip((x - med) / np.maximum(scale, 1e-6), -20, 20).astype(np.float32)


CHB_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3",
                "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
                "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
_ELECTRODE_ALIASES = {"FP1": ("FP1", "FP1"), "FP2": ("FP2", "FP2"),
                      "T7": ("T7", "T3"), "P7": ("P7", "T5"),
                      "T8": ("T8", "T4"), "P8": ("P8", "T6")}


def _clean_channel(name: str) -> str:
    value = name.upper().replace(" ", "").replace(".", "")
    if value.startswith("EEG"):
        value = value[3:]
    for suffix in ("-REF", "-LE", "-AVG", "-CAR"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value


def standardized_montage(raw, requested: int = 18, strict: bool = False) -> tuple[np.ndarray, list[str]]:
    """Return the CHB-MIT bipolar montage, deriving bipolar channels from 10-20 electrodes.

    This lets the optional Siena 10-20 recordings be compared with CHB-MIT only when
    every required derivation can be built. EKG channels and unrelated EEG channels are ignored.
    """
    names = raw.ch_names
    clean = [_clean_channel(name) for name in names]
    index = {name: i for i, name in enumerate(clean)}
    data = raw.get_data()
    output, missing = [], []
    for derivation in CHB_CHANNELS[:requested]:
        left, right = derivation.split("-")
        direct_candidates = [derivation]
        left_alias = _ELECTRODE_ALIASES.get(left, (left,))
        right_alias = _ELECTRODE_ALIASES.get(right, (right,))
        direct_candidates += [f"{l}-{r}" for l in left_alias for r in right_alias]
        direct = next((index[_clean_channel(c)] for c in direct_candidates
                       if _clean_channel(c) in index), None)
        if direct is not None:
            output.append(data[direct])
            continue
        left_idx = next((index[a] for a in left_alias if a in index), None)
        right_idx = next((index[a] for a in right_alias if a in index), None)
        if left_idx is not None and right_idx is not None:
            output.append(data[left_idx] - data[right_idx])
        else:
            missing.append(derivation)
            output.append(np.zeros(data.shape[-1], dtype=np.float64))
    if strict and missing:
        raise ValueError("Cannot standardize this EDF to the required CHB-MIT montage; "
                         f"missing derivations: {', '.join(missing)}. Provide compatible 10-20 channels.")
    return np.asarray(output), missing


def load_recording(edf: Path, summary: Path | None, cfg: dict, horizon: int,
                   seizure_override: list[Seizure] | None = None, strict_channels: bool = False,
                   patient_time_offset: float | None = None):
    import mne
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    signal, missing_channels = standardized_montage(raw, cfg["data"]["channels"], strict_channels)
    fs = float(raw.info["sfreq"])
    d = cfg["data"]
    signal = preprocess(signal, fs, d["sampling_rate"], d["low_hz"], d["high_hz"], d["notch_hz"])
    fs = d["sampling_rate"]
    size = int(d["window_seconds"] * fs)
    step = max(1, int(size * (1 - d["overlap"])))
    patient = re.search(r"chb\d+", edf.name, re.I)
    patient_id = patient.group(0).upper() if patient else edf.parent.name.upper()
    seizures_by_file = parse_summary(summary) if summary and seizure_override is None else {}
    seizures = seizure_override if seizure_override is not None else seizures_by_file.get(edf.name, [])
    time_offset = patient_time_offset or 0.0
    time_reference = "patient_clock" if patient_time_offset is not None else "recording_relative"
    duration_seconds = signal.shape[1] / fs
    for seizure in seizures:
        local_onset = seizure.onset - time_offset if patient_time_offset is not None else seizure.onset
        local_end = seizure.end - time_offset if patient_time_offset is not None else seizure.end
        is_local_event = seizure.recording_id in (None, edf.name)
        if seizure.end <= seizure.onset or (is_local_event and
                (local_onset < 0 or local_end > duration_seconds + 1)):
            raise ValueError(f"Invalid seizure annotation for {edf.name}: {seizure}. "
                             f"Recording duration is {duration_seconds:.1f}s; verify summary time conversion.")
    rows, temporal, spectral, tf = [], [], [], []
    candidates = []
    for wid, start_idx in enumerate(range(0, signal.shape[1] - size + 1, step)):
        start, end = start_idx / fs, (start_idx + size) / fs
        label = label_window(start + time_offset, end + time_offset, seizures, horizon,
                             d["onset_exclusion_seconds"], d["interictal_exclusion_seconds"],
                             d["postictal_exclusion_seconds"])
        candidates.append((wid, start_idx, label))
    # Limit redundant windows per EDF and state for tractable research runs. Evenly spaced
    # selection retains coverage along long interictal and preictal intervals.
    cap = d.get("max_windows_per_recording_per_class", 64)
    selected = []
    for state in (0, 1, 2):
        group = [c for c in candidates if c[2].label == state]
        if cap and len(group) > cap:
            positions = np.linspace(0, len(group) - 1, cap).round().astype(int)
            group = [group[i] for i in positions]
        selected.extend(group)
    # Preserve a complete trace of sampled boundary-ambiguous windows in metadata. They carry
    # the excluded label and are explicitly filtered before every training/evaluation fold.
    selected.extend(c for c in candidates if c[2].label == -1)
    selected.sort(key=lambda c: c[0])
    selected_ids = {c[0] for c in selected}
    all_rows = []
    for wid, start_idx, label in candidates:
        end_idx = start_idx + size
        all_rows.append({"patient_id": patient_id, "recording_id": edf.name, "window_id": wid,
                         "start_time": start_idx / fs, "end_time": end_idx / fs,
                         "patient_start_time": start_idx / fs + time_offset,
                         "patient_end_time": end_idx / fs + time_offset,
                         "time_reference": time_reference,
                         "seizure_id": label.seizure_id, "seizure_onset": label.seizure_onset,
                         "seizure_end": label.seizure_end, "seizure_recording_id": label.seizure_recording_id,
                         "seizure_onset_in_recording": (label.seizure_onset - time_offset
                             if label.seizure_onset is not None and label.seizure_recording_id == edf.name else None),
                         "preictal_horizon": horizon,
                         "exclusion_buffer": d["onset_exclusion_seconds"],
                         "interictal_exclusion_seconds": d["interictal_exclusion_seconds"],
                         "postictal_exclusion_seconds": d["postictal_exclusion_seconds"],
                         "label": label.label, "label_reason": label.reason,
                         "missing_montage_channels": ";".join(missing_channels),
                         "included_in_dataset": wid in selected_ids})
    band_edges = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    for wid, start_idx, label in selected:
        end_idx = start_idx + size
        start, end = start_idx / fs, end_idx / fs
        x = normalize_window(signal[:, start_idx:end_idx])
        # Welch-like mean band power via a Hann-window periodogram.
        freq = np.fft.rfftfreq(size, 1 / fs)
        power = np.abs(np.fft.rfft(x * np.hanning(size)[None, :], axis=-1)) ** 2 / size
        bands = [np.log1p(power[:, (freq >= lo) & (freq < hi)].mean(axis=-1)) for lo, hi in band_edges]
        nperseg = min(64, size)
        _, _, z = stft(x, fs=fs, nperseg=nperseg, noverlap=min(48, size - 1), axis=-1)
        tf_image = np.log1p(np.abs(z)).astype(np.float32)
        # Retain only the configured passband; the temporal dimension comes from the fixed input window.
        keep_bins = min(tf_image.shape[1], int(d["high_hz"] / (fs / nperseg)) + 1)
        tf_image = tf_image[:, :keep_bins, :]
        if tf_image.shape[-1] < 33:
            tf_image = np.pad(tf_image, ((0, 0), (0, 0), (0, 33 - tf_image.shape[-1])))
        temporal.append(x.astype(np.float16))
        spectral.append(np.stack(bands, axis=-1).astype(np.float32))
        tf.append(tf_image.astype(np.float16))
        rows.append({"patient_id": patient_id, "recording_id": edf.name, "window_id": wid,
                     "start_time": start, "end_time": end, "seizure_id": label.seizure_id,
                     "seizure_onset": label.seizure_onset, "seizure_end": label.seizure_end,
                     "patient_start_time": start + time_offset, "patient_end_time": end + time_offset,
                     "time_reference": time_reference, "seizure_recording_id": label.seizure_recording_id,
                     "seizure_onset_in_recording": (label.seizure_onset - time_offset
                         if label.seizure_onset is not None and label.seizure_recording_id == edf.name else None),
                     "preictal_horizon": horizon, "exclusion_buffer": d["onset_exclusion_seconds"],
                     "interictal_exclusion_seconds": d["interictal_exclusion_seconds"],
                     "postictal_exclusion_seconds": d["postictal_exclusion_seconds"],
                     "label": label.label, "label_reason": label.reason, "included_in_dataset": True})
    if not rows:
        return None
    return np.stack(temporal), np.stack(spectral), np.stack(tf), pd.DataFrame(rows), pd.DataFrame(all_rows)


def load_dataset(cfg: dict, horizon: int, metadata_path: str | Path | None = None):
    root = Path(cfg["data"]["root"])
    if not root.exists():
        raise FileNotFoundError(f"CHB-MIT root not found: {root}. Download/extract the EDF dataset first.")
    entries = []
    metadata_path = Path(metadata_path) if metadata_path else None
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.unlink(missing_ok=True)
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summaries = list(patient_dir.glob("*-summary.txt"))
        if not summaries:
            continue
        summary = summaries[0]
        local_events, offsets, patient_events = parse_summary_timeline(summary)
        for edf in sorted(patient_dir.glob("*.edf")):
            if patient_events and edf.name in offsets:
                events = patient_events
                offset = offsets[edf.name]
            else:
                events = local_events.get(edf.name, [])
                offset = None
            data = load_recording(edf, summary, cfg, horizon, seizure_override=events,
                                  patient_time_offset=offset)
            if data is not None:
                entries.append(data[:4])
                if metadata_path and not data[4].empty:
                    data[4].to_csv(metadata_path, index=False, mode="a", header=not metadata_path.exists())
    if not entries:
        raise RuntimeError(f"No usable EDF recordings found under {root}")
    return (np.concatenate([e[0] for e in entries], axis=0),
            np.concatenate([e[1] for e in entries], axis=0),
            np.concatenate([e[2] for e in entries], axis=0),
            pd.concat([e[3] for e in entries], ignore_index=True))
