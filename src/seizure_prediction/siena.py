"""Adapter for the optional PhysioNet Siena Scalp EEG dataset."""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .data import load_recording
from labeling.exclusion import Seizure


def _clock_seconds(value: str) -> float:
    parts = re.split(r"[.:]", value.strip())
    if len(parts) != 3 or not all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
        raise ValueError(f"Unrecognized Siena time (expected hours.minutes.seconds): {value!r}")
    h, m, s = map(float, parts)
    if m >= 60 or s >= 60:
        raise ValueError(f"Invalid hours.minutes.seconds time: {value!r}")
    return h * 3600 + m * 60 + s


def parse_siena_summary(path: Path) -> dict[str, list[Seizure]]:
    """Parse PhysioNet Seizures-list-PNxx.txt registration and event times.

    Event times are converted from clock time to seconds relative to each EDF's
    Registration Start Time. File names and times are read from the released text file.
    """
    result: dict[str, list[Seizure]] = {}
    current_file = None
    registration_start = None
    starts: dict[str, float] = {}
    next_unnumbered = 1
    text = path.read_text(errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        file_match = re.search(r"(?:file\s*name|file)\s*:\s*([^\s,]+\.edf)", line, re.I)
        if file_match:
            if starts:
                raise ValueError(f"Unmatched Siena seizure start time(s) before next file in {path}: {starts}")
            current_file = Path(file_match.group(1)).name
            registration_start = None
            starts = {}
            next_unnumbered = 1
            result.setdefault(current_file, [])
            continue
        if current_file is None:
            continue
        reg = re.search(r"registration\s*start\s*time\s*:\s*([\d.:]+)", line, re.I)
        if reg:
            registration_start = _clock_seconds(reg.group(1))
            continue
        event = re.search(r"seizure\s*(\d*)\s*(start|end)\s*time\s*:\s*([\d.:]+)", line, re.I)
        if not event:
            continue
        idx, boundary, raw_time = event.groups()
        if boundary.lower() == "start" and idx is None:
            idx = str(next_unnumbered)
            next_unnumbered += 1
        seizure_id = idx or (next(reversed(starts)) if starts else None)
        event_clock = _clock_seconds(raw_time)
        if registration_start is None:
            raise ValueError(f"Missing Registration Start Time before seizure entry in {path}: {line}")
        relative = (event_clock - registration_start) % 86400
        if boundary.lower() == "start":
            starts[seizure_id] = relative
        elif seizure_id in starts:
            result[current_file].append(Seizure(starts.pop(seizure_id), relative,
                                                f"{current_file}:sz{seizure_id}", current_file))
        else:
            raise ValueError(f"Seizure end without start in {path}: {line}")
    for filename, events in result.items():
        result[filename] = sorted(events, key=lambda s: s.onset)
    if starts:
        raise ValueError(f"Unmatched Siena seizure start time(s) at end of {path}: {starts}")
    return result


def load_siena_dataset(cfg: dict, horizon: int, metadata_path: str | Path | None = None):
    root = Path(cfg["data"].get("siena_root", "data/siena"))
    if not root.exists():
        raise FileNotFoundError(f"Siena root not found: {root}")
    summaries = list(root.rglob("Seizures-list-PN*.txt"))
    if not summaries:
        raise FileNotFoundError(f"No Seizures-list-PNxx.txt files found under {root}")
    summary_by_dir = {p.parent.resolve(): p for p in summaries}
    out_meta = Path(metadata_path) if metadata_path else None
    if out_meta:
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_meta.unlink(missing_ok=True)
    entries = []
    for edf in sorted(root.rglob("*.edf")):
        summary = summary_by_dir.get(edf.parent.resolve())
        if summary is None:
            raise FileNotFoundError(f"No Siena seizure summary beside {edf}")
        events = parse_siena_summary(summary).get(edf.name)
        if events is None:
            raise ValueError(f"Recording {edf.name} is not described in {summary}")
        strict = not cfg["data"].get("siena_allow_missing_channels", False)
        record = load_recording(edf, None, cfg, horizon, seizure_override=events, strict_channels=strict)
        if record is None:
            continue
        entries.append(record[:4])
        if out_meta and not record[4].empty:
            record[4].to_csv(out_meta, index=False, mode="a", header=not out_meta.exists())
    if not entries:
        raise RuntimeError(f"No usable Siena EDF recordings found under {root}")
    return (np.concatenate([e[0] for e in entries], axis=0),
            np.concatenate([e[1] for e in entries], axis=0),
            np.concatenate([e[2] for e in entries], axis=0),
            pd.concat([e[3] for e in entries], ignore_index=True))


def evaluate_siena(cfg: dict, horizon: int, stage: str, checkpoint_dir: str | Path,
                   dataset=None) -> tuple[pd.DataFrame, dict]:
    """Optional external evaluation using every saved CHB-MIT LOSO fold model."""
    import json
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from .inference import model_from_checkpoint
    from .metrics import patient_metrics, summarize

    if dataset is None:
        metadata = Path(cfg["data"]["output"]) / f"siena_window_labels_{horizon}m.csv"
        dataset = load_siena_dataset(cfg, horizon, metadata)
    raw, spectral, tf, meta = dataset
    valid = meta.label.to_numpy() >= 0
    prediction_valid = valid & (meta.label.to_numpy() != 2)
    ext_patients = sorted(meta.patient_id.unique())
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No CHB-MIT LOSO checkpoints found under {checkpoint_dir}")
    device_cfg = cfg["training"]["device"]
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available()
                          else "cpu" if device_cfg == "auto" else device_cfg)
    outdir = checkpoint_dir / "external_siena"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    arrays = [raw, spectral, tf]
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["horizon"] != horizon or saved["stage"] != stage:
            continue
        model, _ = model_from_checkpoint(saved, device)
        predicted, scores = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(meta), cfg["training"]["batch_size"]):
                batch = [torch.as_tensor(a[start:start + cfg["training"]["batch_size"]],
                                         device=device).float() for a in arrays]
                result = model(*batch)
                predicted.extend(result["state_logits"].argmax(1).cpu().numpy().tolist())
                scores.extend(torch.sigmoid(result["prediction_logits"]).cpu().numpy().tolist())
        fold_name = checkpoint.stem
        for patient in ext_patients:
            pred_ix = np.flatnonzero(prediction_valid & (meta.patient_id.to_numpy() == patient))
            state_ix = np.flatnonzero(valid & (meta.patient_id.to_numpy() == patient))
            if not len(pred_ix) or not len(state_ix):
                continue
            y = (meta.label.to_numpy()[pred_ix] == 1).astype(int)
            pred_metrics = patient_metrics(y, np.asarray(scores)[pred_ix], cfg["training"]["threshold"])
            state_y = meta.label.to_numpy()[state_ix]
            state_pred = np.asarray(predicted)[state_ix]
            rows.append({"siena_patient_id": patient, "chb_loso_patient_excluded": fold_name,
                         "n_prediction_windows": len(pred_ix), "n_state_windows": len(state_ix),
                         "n_excluded_windows": int(((meta.label.to_numpy() < 0) &
                                                     (meta.patient_id.to_numpy() == patient)).sum()),
                         **pred_metrics,
                         "three_state_accuracy": accuracy_score(state_y, state_pred),
                         "three_state_macro_f1": f1_score(state_y, state_pred, average="macro",
                                                           labels=[0, 1, 2], zero_division=0)})
            pd.DataFrame({"patient_id": patient,
                          "recording_id": meta.recording_id.to_numpy()[pred_ix],
                          "window_id": meta.window_id.to_numpy()[pred_ix],
                          "start_time": meta.start_time.to_numpy()[pred_ix],
                          "end_time": meta.end_time.to_numpy()[pred_ix],
                          "true_label": y, "score": np.asarray(scores)[pred_ix],
                          "prediction": np.asarray(scores)[pred_ix] >= cfg["training"]["threshold"]}).to_csv(
                              outdir / f"{fold_name}_{patient}_predictions.csv", index=False)
        del model
    fold_frame = pd.DataFrame(rows)
    if fold_frame.empty:
        raise RuntimeError("No matching checkpoint/results available for Siena evaluation")
    fold_frame.to_csv(outdir / "patient_metrics_by_chb_fold.csv", index=False)
    metric_names = ["sensitivity", "specificity", "f1", "pr_auc", "three_state_accuracy", "three_state_macro_f1"]
    by_patient = fold_frame.groupby("siena_patient_id", as_index=False)[metric_names + ["n_excluded_windows"]].mean()
    by_patient.to_csv(outdir / "siena_patient_metrics.csv", index=False)
    summary = summarize(by_patient.to_dict(orient="records"))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return by_patient, summary
