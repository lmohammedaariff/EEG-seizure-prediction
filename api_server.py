from __future__ import annotations

import os
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "tmp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
for temp_name in ("TEMP", "TMP", "TMPDIR"):
    os.environ[temp_name] = str(UPLOAD_DIR)
tempfile.tempdir = str(UPLOAD_DIR)

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import torch

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from seizure_prediction.config import ExperimentConfig
from seizure_prediction.edf import CHANNELS, preprocess_edf
from seizure_prediction.predict import predict_signal
from seizure_prediction.training import load_checkpoint
from seizure_prediction.cohort import patient_group

DATASET_DIR = ROOT / "data" / "chbmit"
ARTIFACTS_DIR = ROOT / "artifacts" / "results-multihorizon-16case"
DEFAULT_CHECKPOINT = ARTIFACTS_DIR / "final_model" / "model.pt"
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

app = FastAPI(title="Seizure Prediction Research API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _as_python(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _clean_frame(frame: pd.DataFrame) -> list[dict]:
    cleaned = frame.where(pd.notna(frame), None)
    records = cleaned.to_dict(orient="records")
    return [{key: _as_python(value) for key, value in row.items()} for row in records]


def _read_summary_rows() -> list[dict]:
    # Prefer the unique-person aggregates (chb01/chb21 are one individual).
    # Keep a case-level fallback for older result bundles.
    summary_path = ARTIFACTS_DIR / "multihorizon_loso_person_summary.csv"
    if not summary_path.exists():
        summary_path = ARTIFACTS_DIR / "multihorizon_loso_summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    summary = summary.where(pd.notna(summary), None)
    records = summary.to_dict(orient="records")
    return [{key: _as_python(value) for key, value in row.items()} for row in records]


def _read_report_text() -> str:
    report_path = ARTIFACTS_DIR / "final_results_report.md"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8")
    return "No report generated yet."


@lru_cache(maxsize=2)
def _load_model(checkpoint_path: str, checkpoint_mtime_ns: int):
    del checkpoint_mtime_ns
    model, payload, device = load_checkpoint(Path(checkpoint_path), device=torch.device("cpu"))
    config = ExperimentConfig.from_dict(payload["config"])
    return model, config, device


def _analyze_file(
    path: Path, checkpoint_path: Path, horizon_minutes: int, score_threshold: float
):
    model, config, device = _load_model(
        str(checkpoint_path.resolve()),
        checkpoint_path.stat().st_mtime_ns,
    )
    signal, matched_channels = preprocess_edf(path, config)
    frame, windows, attention, domains = predict_signal(
        signal,
        config,
        model,
        device,
        batch_size=128,
        selected_horizon_minutes=horizon_minutes,
        score_threshold=score_threshold,
    )
    return frame, windows, attention, domains, matched_channels, model, device, config


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "project": "Seizure Prediction Research",
        "model_path": str(DEFAULT_CHECKPOINT),
        "model_ready": DEFAULT_CHECKPOINT.is_file(),
        "dataset_path": str(DATASET_DIR),
        "upload_dir": str(UPLOAD_DIR),
    }


@app.get("/api/system-info")
def system_info() -> dict:
    patient_dirs = sorted([path.name for path in DATASET_DIR.iterdir() if path.is_dir()]) if DATASET_DIR.exists() else []
    return {
        "project_root": str(ROOT),
        "dataset_dir": str(DATASET_DIR),
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "patients": patient_dirs,
        "result_dir": str(ARTIFACTS_DIR),
        "report_path": str(ARTIFACTS_DIR / "final_results_report.md"),
    }


@app.get("/api/overview")
def overview() -> dict:
    rows = _read_summary_rows()
    patient_results_path = ARTIFACTS_DIR / "multihorizon_loso_per_patient.csv"
    case_ids: list[str] = []
    if patient_results_path.is_file():
        patient_results = pd.read_csv(
            patient_results_path, usecols=["test_subject"]
        )
        case_ids = sorted(patient_results["test_subject"].dropna().astype(str).unique())
    best_row = None
    if rows:
        numeric_rows = [row for row in rows if row.get("event_sensitivity_mean") is not None]
        if numeric_rows:
            best_row = max(numeric_rows, key=lambda item: float(item["event_sensitivity_mean"]))
    return {
        "total_rows": len(rows),
        "case_count": len(case_ids),
        "people_count": len({patient_group(case_id) for case_id in case_ids}),
        "best_model": best_row,
        "rows": rows,
        "report_preview": _read_report_text()[:1200],
    }


@app.get("/api/reports")
def reports() -> dict:
    return {"report_text": _read_report_text()}


@app.get("/api/recordings")
def recordings() -> dict:
    """List EDFs already present in the local CHB-MIT project folder."""
    items = []
    if DATASET_DIR.is_dir():
        for path in sorted(DATASET_DIR.rglob("*.edf")):
            items.append({
                "path": path.relative_to(DATASET_DIR).as_posix(),
                "patient": path.parent.name.upper(),
                "filename": path.name,
            })
    return {"count": len(items), "recordings": items}


@app.post("/api/analyze-edf")
async def analyze_edf(
    file: UploadFile | None = File(default=None),
    recording: str | None = None,
    checkpoint: str | None = None,
    explain: bool = False,
    include_shap: bool = False,
    horizon_minutes: int = 30,
    score_threshold: float = 0.5,
) -> dict:
    if file is not None and recording:
        raise HTTPException(status_code=400, detail="Choose an uploaded EDF or a local dataset EDF, not both.")
    if file is None and not recording:
        raise HTTPException(status_code=400, detail="Choose a local CHB-MIT EDF or upload an EDF file.")
    temp_path: Path | None = None
    if recording:
        dataset_root = DATASET_DIR.resolve()
        source_path = (dataset_root / recording).resolve()
        if source_path.suffix.lower() != ".edf" or not source_path.is_relative_to(dataset_root):
            raise HTTPException(status_code=400, detail="Recording must be an EDF file inside the local CHB-MIT dataset.")
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="Selected EDF recording was not found in the local dataset.")
        filename = source_path.name
    else:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="EDF file is required.")
        if not file.filename.lower().endswith(".edf"):
            raise HTTPException(status_code=400, detail="Only .edf files are accepted.")
        filename = Path(file.filename).name
    if horizon_minutes not in {5, 10, 15, 30}:
        raise HTTPException(status_code=422, detail="SOP must be 5, 10, 15, or 30 minutes.")
    if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise HTTPException(status_code=422, detail="Score threshold must be between 0 and 1.")

    target_checkpoint = Path(checkpoint).resolve() if checkpoint else DEFAULT_CHECKPOINT.resolve()
    allowed_checkpoint_dir = ARTIFACTS_DIR.resolve()
    if target_checkpoint.suffix.lower() != ".pt" or not target_checkpoint.is_relative_to(
        allowed_checkpoint_dir
    ):
        raise HTTPException(status_code=400, detail="Checkpoint must be a .pt file under the project results directory.")
    if not target_checkpoint.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {target_checkpoint}")

    if file is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_name = f"{uuid.uuid4().hex}_{filename}"
        temp_path = UPLOAD_DIR / temp_name
        bytes_written = 0
        try:
            with temp_path.open("wb") as handle:
                while chunk := file.file.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="EDF uploads must be 256 MB or smaller.",
                        )
                    handle.write(chunk)
        except HTTPException:
            temp_path.unlink(missing_ok=True)
            raise
        source_path = temp_path

    try:
        frame, windows, attention, domains, matched_channels, model, device, config = await run_in_threadpool(
            _analyze_file,
            source_path,
            target_checkpoint,
            horizon_minutes,
            score_threshold,
        )
    except Exception as exc:  # pragma: no cover - surfaced to the UI as an API error
        raise HTTPException(status_code=500, detail=f"EDF analysis failed: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    if frame.empty:
        raise HTTPException(status_code=422, detail="The uploaded EDF did not produce any analyzable windows.")

    top_predictions = frame.nlargest(10, "preictal_probability").copy()
    if horizon_minutes not in config.horizons_minutes:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported horizon. Choose one of {config.horizons_minutes} minutes.",
        )
    explanation = None
    if explain:
        from seizure_prediction.explain import (
            channel_importance,
            integrated_gradients,
            shap_attributions,
            time_frequency_importance,
        )

        top_index = int(frame["preictal_probability"].to_numpy().argmax())
        horizon_index = config.horizons_minutes.index(horizon_minutes)
        input_window = torch.from_numpy(
            np.array(windows[top_index][None], dtype=np.float32, copy=True)
        ).to(device)
        try:
            attribution, predicted_class = await run_in_threadpool(
                integrated_gradients,
                model,
                input_window,
                1,
                16,
                "horizon",
                horizon_index,
            )
            shap_values = None
            if include_shap:
                shap_values = await run_in_threadpool(
                    shap_attributions,
                    model,
                    input_window,
                    1,
                    "horizon",
                    24,
                    horizon_index,
                )
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail="SHAP is optional. Install it with `python -m pip install -e '.[explain]'` and restart the API.",
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

        channel_values = channel_importance(attribution).detach().cpu().numpy()
        time_values = attribution.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
        time_stride = max(1, len(time_values) // 64)
        time_values = time_values[::time_stride][:64]
        tf_freqs, tf_times, tf_map = time_frequency_importance(
            attribution, config.sampling_rate
        )
        valid_freqs = tf_freqs <= 45.0
        tf_map = tf_map[valid_freqs]
        tf_freqs = tf_freqs[valid_freqs]
        tf_scale = float(np.max(tf_map)) if tf_map.size else 0.0
        if tf_scale > 0:
            tf_map = tf_map / tf_scale
        explanation = {
            "method": "Integrated Gradients",
            "target": f"preictal score at the {horizon_minutes}-minute SOP",
            "window_time_minutes": float(frame.iloc[top_index]["time_minutes"]),
            "predicted_class_index": int(predicted_class),
            "channel_importance": [
                {"channel": channel, "importance": float(channel_values[index])}
                for index, channel in enumerate(CHANNELS)
                if channel in matched_channels
            ],
            "time_importance": [float(value) for value in time_values],
            "time_importance_seconds": [
                float(value)
                for value in np.linspace(
                    0, config.window_seconds, len(time_values), endpoint=False
                )
            ],
            "time_frequency": {
                "frequencies_hz": [float(value) for value in tf_freqs],
                "times_seconds": [float(value) for value in tf_times],
                "importance": tf_map.astype(float).tolist(),
            },
            "shap_channel_importance": (
                [
                    {"channel": channel, "importance": float(value)}
                    for channel, value in zip(
                        CHANNELS,
                        channel_importance(shap_values).detach().cpu().numpy(),
                    )
                    if channel in matched_channels
                ]
                if shap_values is not None
                else None
            ),
            "attention_by_domain": (
                {
                    name: float(attention[top_index, index])
                    for index, name in enumerate(domains)
                }
                if attention is not None and len(attention) > top_index
                else {}
            ),
        }
    summary = {
        "filename": filename,
        "window_count": int(len(frame)),
        "prediction_rows": int(len(frame)),
        "channel_count": int(len(matched_channels)),
        "channels": list(matched_channels[:12]),
        "mean_preictal_probability": float(frame["preictal_probability"].mean()),
        "max_preictal_probability": float(frame["preictal_probability"].max()),
        "prediction_horizon_minutes": horizon_minutes,
        "score_threshold": score_threshold,
        "prediction_counts": {
            label: int((frame["prediction"] == label).sum()) for label in ["Interictal", "Preictal"]
        },
        "peak_prediction": top_predictions.iloc[0].to_dict() if not top_predictions.empty else None,
    }
    # Keep a representative sample from the whole EDF for the risk chart,
    # rather than returning only the beginning of long recordings.
    stride = max(1, int(np.ceil(len(frame) / 600)))
    plot_frame = frame.iloc[::stride].copy()
    if len(frame) and (len(plot_frame) == 0 or plot_frame.index[-1] != frame.index[-1]):
        plot_frame = pd.concat([plot_frame, frame.iloc[[-1]]])
    plot_columns = [
        column
        for column in frame.columns
        if column in {"time_minutes", "prediction", "preictal_probability"}
        or column.startswith("preictal_probability_sop_")
    ]

    # The model input is channel-standardized per window. Plot a few real
    # channels from the highest-scoring window on a shared, normalized scale.
    peak_index = int(frame["preictal_probability"].to_numpy().argmax())
    sample_stride = max(1, int(np.ceil(windows.shape[-1] / 320)))
    trace_channels = []
    for channel in matched_channels[:4]:
        channel_index = CHANNELS.index(channel)
        values = windows[peak_index, channel_index, ::sample_stride]
        trace_channels.append({"channel": channel, "values": values.astype(float).tolist()})
    eeg_trace = {
        "window_start_minutes": float(frame.iloc[peak_index]["time_minutes"])
        - config.window_seconds / 120,
        "window_seconds": float(config.window_seconds),
        "sampling_rate_hz": float(config.sampling_rate / sample_stride),
        "amplitude_label": "per-channel z-score",
        "channels": trace_channels,
    }
    selected_channel_indices = [CHANNELS.index(channel) for channel in matched_channels[:4]]
    trace_window = windows[peak_index, selected_channel_indices, :].astype(np.float64)
    sample_count = trace_window.shape[-1]
    tapered = trace_window * np.hanning(sample_count)[None, :]
    spectrum = np.mean(np.abs(np.fft.rfft(tapered, axis=1)) ** 2, axis=0)
    spectrum /= max(float(spectrum.max()), 1e-12)
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / config.sampling_rate)
    keep_frequency = frequencies <= 45.0
    spectral_stride = max(1, int(np.ceil(int(keep_frequency.sum()) / 160)))

    first_channel = trace_window[0]
    frame_length = min(128, sample_count)
    hop_length = max(1, frame_length // 4)
    frames = np.lib.stride_tricks.sliding_window_view(first_channel, frame_length)[::hop_length]
    tf_magnitude = np.abs(
        np.fft.rfft(frames * np.hanning(frame_length)[None, :], n=256, axis=1)
    ).T
    tf_frequencies = np.fft.rfftfreq(256, d=1.0 / config.sampling_rate)
    keep_tf_frequency = tf_frequencies <= 45.0
    tf_magnitude = tf_magnitude[keep_tf_frequency]
    tf_min = float(tf_magnitude.min()) if tf_magnitude.size else 0.0
    tf_max = float(tf_magnitude.max()) if tf_magnitude.size else 0.0
    tf_magnitude = (tf_magnitude - tf_min) / max(tf_max - tf_min, 1e-12)
    tf_time_stride = max(1, int(np.ceil(tf_magnitude.shape[1] / 64)))
    tf_frequency_stride = max(1, int(np.ceil(tf_magnitude.shape[0] / 72)))
    signal_transforms = {
        "spectrum": {
            "frequencies_hz": frequencies[keep_frequency][::spectral_stride].astype(float).tolist(),
            "relative_power": spectrum[keep_frequency][::spectral_stride].astype(float).tolist(),
        },
        "spectrogram": {
            "frequencies_hz": tf_frequencies[keep_tf_frequency][::tf_frequency_stride].astype(float).tolist(),
            "times_seconds": (np.arange(frames.shape[0]) * hop_length / config.sampling_rate + frame_length / (2 * config.sampling_rate))[::tf_time_stride].astype(float).tolist(),
            "relative_power": tf_magnitude[::tf_frequency_stride, ::tf_time_stride].astype(float).tolist(),
        },
    }

    return {
        "summary": summary,
        "selected_horizon_minutes": horizon_minutes,
        "score_threshold": score_threshold,
        "top_predictions": _clean_frame(top_predictions),
        "predictions": _clean_frame(frame.head(50)),
        "plot_data": _clean_frame(plot_frame[plot_columns]),
        "eeg_trace": eeg_trace,
        "signal_transforms": signal_transforms,
        "matched_channels": list(matched_channels),
        "attention_domains": domains,
        "explanation": explanation,
    }


FRONTEND_DIST = ROOT / "frontend" / "dist"
if (FRONTEND_DIST / "index.html").is_file():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="react-assets")

    @app.get("/{frontend_path:path}", include_in_schema=False)
    async def serve_react_app(frontend_path: str) -> FileResponse:
        # The React client owns navigation. Return its entry point for routes
        # (and malformed paths such as /**) rather than a StaticFiles 500.
        if frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
