from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from seizure_prediction.config import CHANNELS, ExperimentConfig
from seizure_prediction.datasets import MultiHorizonEEGDataset
from seizure_prediction.edf import relabel_cache_horizons
from seizure_prediction.event_metrics import (
    calibrate_thresholds,
    event_forecast_metrics,
)
from seizure_prediction.multihorizon_eval import evaluate_multihorizon_loso
from seizure_prediction.models import build_model
from seizure_prediction.training import (
    predict_multihorizon_loader,
    train_final_multihorizon_model,
    train_multihorizon_model,
)


def _make_multihorizon_cache(
    root: Path,
    patient_id: str,
    horizons: tuple[int, ...] = (5, 10),
    *,
    n_windows: int = 80,
    step_seconds: float = 5.0,
    event_onset: float = 200.0,
) -> None:
    patient = root / patient_id
    patient.mkdir(parents=True)
    config = ExperimentConfig(
        sampling_rate=128,
        window_seconds=10,
        overlap=0.5,
        horizons_minutes=horizons,
        preictal_minutes=max(horizons),
        sph_minutes=0,
        interictal_buffer_minutes=0,
        epochs=1,
        batch_size=8,
    )
    rng = np.random.default_rng(sum(patient_id.encode("ascii")))
    windows = rng.normal(size=(n_windows, len(CHANNELS), 128)).astype(np.float32)
    starts = np.arange(n_windows, dtype=np.float32) * step_seconds
    np.save(patient / "record.x.npy", windows)
    np.save(patient / "record.t.npy", starts)
    y_by_horizon = {}
    counts_by_horizon = {}
    for horizon in horizons:
        warning_start = event_onset - horizon * 60.0
        labels = np.zeros(n_windows, dtype=np.int8)
        labels[(starts >= warning_start) & (starts + 10 <= event_onset)] = 1
        labels[(starts < event_onset + 20) & (starts + 10 > event_onset)] = 2
        labels[starts >= event_onset + 20] = 0
        y_name = f"record.y-h{horizon}.npy"
        np.save(patient / y_name, labels)
        y_by_horizon[str(horizon)] = y_name
        counts_by_horizon[str(horizon)] = np.bincount(labels, minlength=3).tolist()
    manifest = {
        "patient_id": patient_id,
        "config": config.to_dict(),
        "recording_offsets_seconds": {"record.edf": 0},
        "seizures_by_recording": {"record.edf": [[event_onset, event_onset + 20]]},
        "records": [
            {
                "recording": "record.edf",
                "x": "record.x.npy",
                "t": "record.t.npy",
                "y_by_horizon": y_by_horizon,
                "windows": n_windows,
                "class_counts_by_horizon": counts_by_horizon,
            }
        ],
    }
    (patient / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_multihorizon_model_returns_horizon_specific_logits() -> None:
    variants = (
        "multi_horizon_temporal",
        "multi_horizon_temporal_spectral",
        "multi_horizon_multi_domain",
        "multi_horizon",
        "multi_horizon_cnn",
        "multi_horizon_lstm",
        "multi_horizon_cnn_lstm",
    )
    for variant in variants:
        model = build_model(
            n_channels=4,
            n_subjects=3,
            sampling_rate=128,
            variant=variant,
            horizons_minutes=(5, 10, 30),
        )
        output = model(torch.randn(2, 4, 128))
        assert output["state_logits"].shape == (2, 3)
        assert output["horizon_logits"].shape == (2, 3, 2)
        if variant == "multi_horizon":
            assert output["subject_logits"].shape == (2, 3)
        elif variant.startswith("multi_horizon_") and variant not in {
            "multi_horizon_cnn",
            "multi_horizon_lstm",
            "multi_horizon_cnn_lstm",
        }:
            assert output["subject_logits"] is None
            assert model.use_spectral == (
                variant
                in {
                    "multi_horizon_temporal_spectral",
                    "multi_horizon_multi_domain",
                }
            )
            assert model.use_time_frequency == (
                variant == "multi_horizon_multi_domain"
            )


def test_multihorizon_dataset_keeps_horizon_masks_and_record_times(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _make_multihorizon_cache(cache, "chb01")
    dataset = MultiHorizonEEGDataset(cache / "chb01", (5, 10), subject_index=2)

    sample = dataset[0]

    assert sample[0].shape == (len(CHANNELS), 128)
    assert sample[2].shape == (2,)
    assert sample[3].item() == 2
    assert sample[4].item() == 0
    assert sample[5].item() == pytest.approx(0.0)
    assert dataset.horizon_class_counts.shape == (2, 2)
    assert dataset.seizures_by_recording["record.edf"] == [(200.0, 220.0)]


def test_relabel_cache_adds_all_requested_sops_without_edf_preprocessing(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    patient_dir = data_dir / "chb01"
    patient_cache = cache_dir / "chb01"
    patient_dir.mkdir(parents=True)
    patient_cache.mkdir(parents=True)
    config = ExperimentConfig(
        horizons_minutes=(30,),
        preictal_minutes=30,
        sph_minutes=5,
        interictal_buffer_minutes=0,
    )
    starts = np.arange(0, 4000, 5, dtype=np.float32)
    windows = np.zeros((len(starts), 2, 16), dtype=np.float32)
    np.save(patient_cache / "chb01_01.x.npy", windows)
    np.save(patient_cache / "chb01_01.t.npy", starts)
    np.save(patient_cache / "chb01_01.y-h30.npy", np.zeros(len(starts), dtype=np.int8))
    summary = (
        "File Name: chb01_01.edf\n"
        "File Start Time: 00:00:00\n"
        "Number of Seizures in File: 1\n"
        "Seizure 1 Start Time: 3000 seconds\n"
        "Seizure 1 End Time: 3020 seconds\n"
    )
    (patient_dir / "chb01-summary.txt").write_text(summary, encoding="utf-8")
    manifest = {
        "patient_id": "chb01",
        "config": config.to_dict(),
        "recording_offsets_seconds": {"chb01_01.edf": 0},
        "records": [
            {
                "recording": "chb01_01.edf",
                "x": "chb01_01.x.npy",
                "t": "chb01_01.t.npy",
                "y_by_horizon": {"30": "chb01_01.y-h30.npy"},
                "windows": len(starts),
            }
        ],
    }
    manifest_path = patient_cache / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    relabel_cache_horizons(data_dir, cache_dir, (5, 10, 15, 30))

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["config"]["horizons_minutes"] == [5, 10, 15, 30]
    assert updated["seizures_by_recording"]["chb01_01.edf"] == [[3000.0, 3020.0]]
    assert set(updated["records"][0]["y_by_horizon"]) == {"5", "10", "15", "30"}
    assert all(
        np.any(np.load(patient_cache / name) == 1)
        for name in updated["records"][0]["y_by_horizon"].values()
    )


def test_relabel_cache_rejects_summary_recordings_without_cached_offsets(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    patient_dir = data_dir / "chb01"
    patient_cache = cache_dir / "chb01"
    patient_dir.mkdir(parents=True)
    patient_cache.mkdir(parents=True)
    config = ExperimentConfig(horizons_minutes=(30,))
    starts = np.arange(0, 100, 5, dtype=np.float32)
    np.save(patient_cache / "chb01_01.x.npy", np.zeros((20, 2, 16), dtype=np.float32))
    np.save(patient_cache / "chb01_01.t.npy", starts)
    np.save(patient_cache / "chb01_01.y-h30.npy", np.zeros(20, dtype=np.int8))
    (patient_dir / "chb01-summary.txt").write_text(
        "File Name: chb01_01.edf\n"
        "Number of Seizures in File: 0\n"
        "File Name: chb01_02.edf\n"
        "Number of Seizures in File: 1\n"
        "Seizure 1 Start Time: 50 seconds\n"
        "Seizure 1 End Time: 60 seconds\n",
        encoding="utf-8",
    )
    manifest = {
        "patient_id": "chb01",
        "config": config.to_dict(),
        "recording_offsets_seconds": {"chb01_01.edf": 0},
        "records": [
            {
                "recording": "chb01_01.edf",
                "x": "chb01_01.x.npy",
                "t": "chb01_01.t.npy",
                "y_by_horizon": {"30": "chb01_01.y-h30.npy"},
                "windows": len(starts),
            }
        ],
    }
    (patient_cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="summary references EDF recordings"):
        relabel_cache_horizons(data_dir, cache_dir, (5, 10, 15, 30))


def test_event_metrics_count_seizures_and_merged_alarm_episodes() -> None:
    records = {
        "record.edf": __import__("pandas").DataFrame(
            {
                "time_seconds": [0, 100, 400, 405, 410, 1000],
                "label": [0, 0, 1, 1, 1, 0],
                "score": [0.95, 0.1, 0.90, 0.92, 0.93, 0.1],
            }
        )
    }

    result = event_forecast_metrics(
        records,
        {"record.edf": 0.0},
        [(1200.0, 1220.0)],
        horizon_minutes=10,
        sph_minutes=5,
        threshold=0.5,
        window_seconds=10,
        alarm_merge_gap_minutes=1,
    )

    assert result["eligible_events"] == 1
    assert result["detected_events"] == 1
    assert result["event_sensitivity"] == pytest.approx(1.0)
    assert result["alarm_count"] == 2
    assert result["false_alarms"] == 1
    assert result["mean_lead_time_minutes"] == pytest.approx(790 / 60)


def test_one_alarm_cannot_detect_two_overlapping_seizure_forecasts() -> None:
    import pandas as pd

    records = {
        "record.edf": pd.DataFrame(
            {"time_seconds": [660], "label": [1], "score": [0.9]}
        )
    }
    result = event_forecast_metrics(
        records,
        {"record.edf": 0.0},
        [(700.0, 710.0), (720.0, 730.0)],
        horizon_minutes=1,
        sph_minutes=0,
        threshold=0.5,
        window_seconds=5.0,
        alarm_merge_gap_minutes=0,
    )

    assert result["alarm_count"] == 1
    assert result["detected_events"] == 1
    assert result["event_sensitivity"] == pytest.approx(0.5)


def test_interictal_exposure_unions_overlapping_windows_and_gaps() -> None:
    import pandas as pd

    result = event_forecast_metrics(
        {
            "record.edf": pd.DataFrame(
                {
                    "time_seconds": [0, 5, 50],
                    "label": [0, 0, 0],
                    "score": [0.1, 0.1, 0.1],
                }
            )
        },
        {"record.edf": 0.0},
        [],
        horizon_minutes=5,
        sph_minutes=0,
        threshold=0.5,
        window_seconds=10,
        alarm_merge_gap_minutes=0,
    )

    assert result["eligible_interictal_hours"] == pytest.approx(25 / 3600)
    assert result["false_alarms_per_hour"] == 0


def test_threshold_is_selected_on_validation_at_alarm_budget() -> None:
    import pandas as pd

    records = {
        "record.edf": pd.DataFrame(
            {
                "time_seconds": [0, 10, 20, 30, 40, 400, 405],
                "label": [0, 0, 0, 0, 0, 1, 1],
                "score": [0.9, 0.1, 0.1, 0.1, 0.1, 0.8, 0.85],
            }
        )
    }
    selected = calibrate_thresholds(
        records,
        {"record.edf": 0.0},
        [(500.0, 520.0)],
        horizon_minutes=5,
        sph_minutes=0,
        false_alarm_budgets_per_hour=(0.0, 100.0),
        window_seconds=10,
        alarm_merge_gap_minutes=1,
    )

    assert selected[0.0][1]["false_alarms_per_hour"] == 0
    assert selected[100.0][1]["event_sensitivity"] == 1.0


@pytest.mark.parametrize("model_variant", ["multi_horizon_temporal", "multi_horizon_cnn"])
def test_multihorizon_training_saves_loadable_checkpoint(
    tmp_path: Path,
    model_variant: str,
) -> None:
    cache = tmp_path / "cache"
    _make_multihorizon_cache(
        cache,
        "chb01",
        horizons=(5, 10),
        n_windows=24,
        step_seconds=5,
        event_onset=60,
    )
    _make_multihorizon_cache(
        cache,
        "chb02",
        horizons=(5, 10),
        n_windows=24,
        step_seconds=5,
        event_onset=60,
    )
    train_data = [
        MultiHorizonEEGDataset(cache / "chb01", (5, 10), subject_index=0),
        MultiHorizonEEGDataset(cache / "chb02", (5, 10), subject_index=1),
    ]
    validation = MultiHorizonEEGDataset(cache / "chb01", (5, 10), subject_index=2)
    config = ExperimentConfig(
        horizons_minutes=(5, 10),
        preictal_minutes=10,
        sph_minutes=0,
        epochs=1,
        batch_size=8,
        model_variant=model_variant,
    )
    checkpoint = tmp_path / "model.pt"

    result = train_multihorizon_model(
        train_data,
        validation,
        config,
        model_variant,
        checkpoint,
        samples_per_epoch=8,
    )

    assert result["best_epoch"] == 1
    model = build_model(
        n_channels=len(CHANNELS),
        n_subjects=2,
        sampling_rate=config.sampling_rate,
        variant=model_variant,
        horizons_minutes=(5, 10),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    predictions = predict_multihorizon_loader(
        model,
        DataLoader(validation, batch_size=8, shuffle=False),
        torch.device("cpu"),
    )
    assert predictions["horizon_probabilities"].shape == (len(validation), 2)


def test_final_multihorizon_fit_uses_all_windows_and_saves_checkpoint(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    for patient in ("chb01", "chb02"):
        _make_multihorizon_cache(
            cache,
            patient,
            horizons=(5, 10),
            n_windows=24,
            step_seconds=5,
            event_onset=60,
        )
    datasets = [
        MultiHorizonEEGDataset(cache / patient, (5, 10), subject_index=index)
        for index, patient in enumerate(("chb01", "chb02"))
    ]
    config = ExperimentConfig(
        horizons_minutes=(5, 10),
        preictal_minutes=10,
        sph_minutes=0,
        epochs=1,
        batch_size=8,
    )
    checkpoint = tmp_path / "final" / "model.pt"

    result = train_final_multihorizon_model(
        datasets,
        config,
        "multi_horizon_temporal",
        checkpoint,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert result["training_windows"] == sum(map(len, datasets))
    assert result["training_recordings"] == sum(
        len(dataset.recording_names) for dataset in datasets
    )
    assert result["device"] == "cpu"
    assert payload["validation_subject"] is None
    assert payload["training_samples_per_epoch"] == sum(map(len, datasets))
    assert "not an independent evaluation" in payload["checkpoint_protocol"]


def test_multihorizon_loso_resume_keeps_complete_folds_without_duplicates(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    for index in range(1, 5):
        _make_multihorizon_cache(
            cache,
            f"chb{index:02d}",
            horizons=(5, 10),
            n_windows=24,
            step_seconds=5,
            event_onset=60,
        )
    config = ExperimentConfig(
        horizons_minutes=(5, 10),
        preictal_minutes=10,
        sph_minutes=0,
        interictal_buffer_minutes=0,
        epochs=1,
        batch_size=8,
        model_variant="multi_horizon_cnn",
    )
    output = tmp_path / "results"
    kwargs = {
        "cache_dir": cache,
        "output_dir": output,
        "config": config,
        "model_variants": ("multi_horizon_cnn",),
        "samples_per_epoch": 8,
        "threshold_candidates": 3,
    }

    first = evaluate_multihorizon_loso(**kwargs, max_folds=1)
    resumed = evaluate_multihorizon_loso(
        **kwargs, max_folds=2, resume=True
    )

    assert len(first) == 6
    assert len(resumed) == 12
    assert resumed.groupby(["model_variant", "test_subject"]).size().to_dict() == {
        ("multi_horizon_cnn", "chb01"): 6,
        ("multi_horizon_cnn", "chb02"): 6,
    }
