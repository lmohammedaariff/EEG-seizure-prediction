import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from seizure_prediction.config import CHANNELS, ExperimentConfig
from seizure_prediction.datasets import HorizonEEGDataset
import seizure_prediction.edf as edf_module
from seizure_prediction.edf import prepare_dataset, standardize_channels, window_recording
from seizure_prediction.evaluation import calculate_metrics, evaluate_loso
from seizure_prediction.explain import (
    integrated_gradients,
    shap_attributions,
    time_frequency_importance,
)
from seizure_prediction.labels import INTERICTAL, Seizure
from seizure_prediction.models import (
    BaselineEEGNet,
    MultiDomainEEGNet,
    gradient_reverse,
)


def test_standardize_channels_aliases_and_zero_fills() -> None:
    source = np.ones((4, 64), dtype=np.float32)
    standardized, matched = standardize_channels(
        source,
        ["EEG FP1-F7-LE", "F7-T3", "T3-T5", "T5-O1"],
    )
    assert standardized.shape == (len(CHANNELS), 64)
    assert matched[:4] == ["FP1-F7", "F7-T7", "T7-P7", "P7-O1"]
    assert np.any(standardized[0])


def test_windowing_uses_configured_overlap_and_normalizes_channels() -> None:
    config = ExperimentConfig(
        sampling_rate=128,
        window_seconds=1,
        overlap=0.5,
        preictal_minutes=5,
        sph_minutes=0,
        interictal_buffer_minutes=0,
        horizons_minutes=(5,),
    )
    t = np.arange(320) / 128
    signal = np.stack([np.sin(2 * np.pi * 8 * t), np.zeros_like(t)]).astype(np.float32)
    windows, labels, starts = window_recording(signal, [], config)
    assert windows.shape == (4, 2, 128)
    assert labels.tolist() == [INTERICTAL] * 4
    assert starts.tolist() == pytest.approx([0, 0.5, 1.0, 1.5])
    assert windows[:, 1].max() == 0
    assert np.isfinite(windows).all()
    assert windows[:, 0].std(axis=-1) == pytest.approx(np.ones(4), abs=0.02)


def test_prepare_dataset_ignores_incomplete_download_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    (data / "chb01").mkdir(parents=True)
    (data / "chb01" / "chb01-summary.txt").touch()
    (data / "chb01" / "chb01_01.edf.part").touch()
    (data / "chb02").mkdir()
    (data / "chb02" / "chb02_01.edf").touch()
    (data / "chb03").mkdir()
    (data / "chb03" / "chb03_01.edf").touch()
    (data / "chb03" / "chb03_02.edf.part").touch()
    prepared: list[str] = []

    def record_prepared_patient(
        patient_dir: Path,
        cache_dir: Path,
        config: ExperimentConfig,
        max_recordings: int | None = None,
    ) -> dict[str, object]:
        prepared.append(patient_dir.name)
        return {"patient_id": patient_dir.name}

    monkeypatch.setattr(edf_module, "prepare_patient", record_prepared_patient)

    manifests = prepare_dataset(
        data,
        tmp_path / "cache",
        ExperimentConfig(horizons_minutes=(30,)),
    )

    assert prepared == ["chb02"]
    assert manifests == [{"patient_id": "chb02"}]


@pytest.mark.parametrize(
    ("variant", "domains"),
    [
        ("temporal_only", 1),
        ("temporal_spectral", 2),
        ("multi_domain", 3),
        ("subject_invariant", 3),
    ],
)
def test_multidomain_model_returns_state_prediction_and_attention(
    variant: str, domains: int
) -> None:
    model = MultiDomainEEGNet(n_channels=4, n_subjects=3, variant=variant)
    result = model(torch.randn(2, 4, 128))
    assert result["state_logits"].shape == (2, 3)
    assert result["prediction_logits"].shape == (2, 2)
    assert result["attention"].shape == (2, domains)
    assert torch.allclose(
        result["attention"].sum(dim=1), torch.ones(2), atol=1e-6
    )
    if variant == "subject_invariant":
        assert result["subject_logits"].shape == (2, 3)


@pytest.mark.parametrize("variant", ["cnn", "lstm", "cnn_lstm"])
def test_baseline_models_produce_both_task_outputs(variant: str) -> None:
    model = BaselineEEGNet(n_channels=4, variant=variant)
    result = model(torch.randn(2, 4, 128))
    assert result["state_logits"].shape == (2, 3)
    assert result["prediction_logits"].shape == (2, 2)
    assert result["attention"] is None


def test_gradient_reversal_changes_gradient_sign() -> None:
    value = torch.tensor([2.0], requires_grad=True)
    gradient_reverse(value, 0.5).sum().backward()
    assert value.grad.item() == pytest.approx(-0.5)


def test_integrated_gradients_returns_finite_input_attributions() -> None:
    model = MultiDomainEEGNet(
        n_channels=4,
        n_subjects=0,
        variant="temporal_only",
    )
    attribution, predicted = integrated_gradients(
        model, torch.randn(1, 4, 128), steps=3
    )
    assert attribution.shape == (1, 4, 128)
    assert predicted in (0, 1, 2)
    assert torch.isfinite(attribution).all()


def test_time_frequency_map_has_valid_axes_and_finite_values() -> None:
    attribution = torch.randn(1, 4, 128)
    frequencies, times, importance = time_frequency_importance(
        attribution, sampling_rate=128
    )
    assert importance.shape == (len(frequencies), len(times))
    assert np.isfinite(importance).all()


@pytest.mark.skipif(
    importlib.util.find_spec("shap") is None,
    reason="Install the optional explain extra to run GradientSHAP tests.",
)
def test_shap_attributions_match_input_shape() -> None:
    model = MultiDomainEEGNet(n_channels=4, variant="temporal_only").eval()
    signal = torch.randn(1, 4, 128)
    values = shap_attributions(model, signal, target_class=1, nsamples=4)
    assert values.shape == signal.shape
    assert torch.isfinite(values).all()


def test_prediction_metrics_keep_state_and_binary_tasks_separate() -> None:
    y_true = np.array([0, 1, 2, 0])
    state_probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
        ]
    )
    prediction_probs = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]]
    )
    metrics = calculate_metrics(
        y_true,
        state_probs.argmax(axis=1),
        state_probs,
        prediction_probs,
    )
    assert metrics["state_accuracy"] == pytest.approx(1.0)
    assert metrics["prediction_sensitivity"] == pytest.approx(1.0)
    assert metrics["prediction_specificity"] == pytest.approx(1.0)
    assert metrics["prediction_pr_auc"] == pytest.approx(1.0)


def _write_subject_cache(
    root: Path,
    subject: str,
    horizon: int,
    rng: np.random.Generator,
) -> None:
    patient = root / subject
    patient.mkdir(parents=True)
    labels = np.tile(np.array([0, 1, 2], dtype=np.int8), 4)
    windows = rng.normal(size=(len(labels), len(CHANNELS), 128)).astype(np.float32)
    np.save(patient / "record.x.npy", windows)
    np.save(patient / f"record.y-h{horizon}.npy", labels)
    np.save(patient / "record.t.npy", np.arange(len(labels), dtype=np.float32))
    manifest = {
        "patient_id": subject,
        "config": ExperimentConfig(
            sampling_rate=128,
            window_seconds=1,
            overlap=0.5,
            preictal_minutes=horizon,
            sph_minutes=0,
            horizons_minutes=(horizon,),
            interictal_buffer_minutes=0,
        ).to_dict(),
        "channels": list(CHANNELS),
        "records": [
            {
                "recording": "record.edf",
                "x": "record.x.npy",
                "y_by_horizon": {str(horizon): f"record.y-h{horizon}.npy"},
                "t": "record.t.npy",
                "windows": len(labels),
                "class_counts_by_horizon": {str(horizon): [4, 4, 4]},
                "matched_channels": list(CHANNELS),
            }
        ],
    }
    (patient / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_loso_smoke_trains_saves_loadable_model_and_metrics(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(2)
    for index in range(4):
        _write_subject_cache(cache, f"chb{index + 1:02d}", 5, rng)
    results = evaluate_loso(
        cache_dir=cache,
        output_dir=tmp_path / "results",
        config=ExperimentConfig(
            sampling_rate=128,
            window_seconds=1,
            overlap=0.5,
            horizons_minutes=(5,),
            preictal_minutes=5,
            sph_minutes=0,
            interictal_buffer_minutes=0,
            batch_size=8,
            epochs=1,
            model_variant="temporal_only",
            random_seed=1,
        ),
        model_variants=("temporal_only", "subject_invariant"),
        horizons=(5,),
        max_folds=1,
    )
    assert len(results) == 2
    assert (tmp_path / "results" / "loso_per_subject.csv").is_file()
    assert (tmp_path / "results" / "loso_summary.csv").is_file()
    checkpoint = Path(results.iloc[0]["checkpoint"])
    assert checkpoint.is_file()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert loaded["model_variant"] == "temporal_only"
    assert loaded["config"]["preictal_minutes"] == 5
    assert len(HorizonEEGDataset(cache / "chb01", 5)) == 12
    invariant = results.loc[results["model_variant"] == "subject_invariant"].iloc[0]
    assert Path(invariant["checkpoint"]).is_file()
