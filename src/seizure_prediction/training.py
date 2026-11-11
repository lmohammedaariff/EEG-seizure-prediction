from __future__ import annotations

import math
import os
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .config import CHANNELS, MULTIHORIZON_MODEL_VARIANTS, ExperimentConfig
from .datasets import HorizonEEGDataset, MultiHorizonEEGDataset
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_train_loader(
    dataset: ConcatDataset,
    batch_size: int,
    samples_per_epoch: int | None = None,
    replacement: bool = True,
) -> DataLoader:
    subject_weights = []
    n_subjects = len(dataset.datasets)
    for patient_data in dataset.datasets:
        counts = patient_data.class_counts
        present = counts > 0
        class_weights = np.zeros(3, dtype=np.float64)
        class_weights[present] = 1 / (present.sum() * counts[present])
        subject_weights.append(class_weights[patient_data.labels] / n_subjects)
    sample_weights = np.concatenate(subject_weights)
    if samples_per_epoch is not None and samples_per_epoch <= 0:
        raise ValueError("Samples per epoch must be positive when provided.")
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=(
            len(sample_weights)
            if samples_per_epoch is None
            else min(samples_per_epoch, len(sample_weights))
        ),
        replacement=replacement,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def _make_eval_loader(dataset: HorizonEEGDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    prediction_probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for signals, states, _ in loader:
            output = model(signals.to(device))
            logits = output["state_logits"]
            prediction_logits = output["prediction_logits"]
            if not isinstance(logits, torch.Tensor) or not isinstance(
                prediction_logits, torch.Tensor
            ):
                raise TypeError("The model did not return state logits.")
            probs = torch.softmax(logits, dim=1)
            prediction_probs = torch.softmax(prediction_logits, dim=1)
            labels.append(states.numpy())
            predictions.append(probs.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            prediction_probabilities.append(prediction_probs.cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(probabilities),
        np.concatenate(prediction_probabilities),
    )


def predict_multihorizon_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Collect state and horizon-head scores without mixing recording timelines."""
    model.eval()
    states: list[np.ndarray] = []
    horizon_labels: list[np.ndarray] = []
    state_predictions: list[np.ndarray] = []
    state_probabilities: list[np.ndarray] = []
    horizon_probabilities: list[np.ndarray] = []
    recording_indices: list[np.ndarray] = []
    starts_seconds: list[np.ndarray] = []
    with torch.inference_mode():
        for signals, labels, targets, _, record_ids, starts in loader:
            output = model(signals.to(device))
            state_logits = output["state_logits"]
            horizon_logits = output["horizon_logits"]
            if not isinstance(state_logits, torch.Tensor) or not isinstance(
                horizon_logits, torch.Tensor
            ):
                raise TypeError("Multi-horizon model did not return all required logits.")
            states.append(labels.numpy())
            horizon_labels.append(targets.numpy())
            state_probabilities.append(torch.softmax(state_logits, dim=1).cpu().numpy())
            state_predictions.append(state_logits.argmax(dim=1).cpu().numpy())
            horizon_probabilities.append(
                torch.softmax(horizon_logits, dim=2)[:, :, 1].cpu().numpy()
            )
            recording_indices.append(record_ids.numpy())
            starts_seconds.append(starts.numpy())
    return {
        "state_labels": np.concatenate(states),
        "horizon_labels": np.concatenate(horizon_labels),
        "state_predictions": np.concatenate(state_predictions),
        "state_probabilities": np.concatenate(state_probabilities),
        "horizon_probabilities": np.concatenate(horizon_probabilities),
        "recording_indices": np.concatenate(recording_indices),
        "starts_seconds": np.concatenate(starts_seconds),
    }


def train_multihorizon_model(
    train_datasets: list[MultiHorizonEEGDataset],
    validation_dataset: MultiHorizonEEGDataset,
    config: ExperimentConfig,
    model_variant: str,
    checkpoint_path: Path,
    samples_per_epoch: int | None = None,
) -> dict[str, object]:
    """Train shared features with a separate, mask-aware head for every SOP."""
    if not train_datasets:
        raise ValueError("At least one training patient is required.")
    if tuple(train_datasets[0].horizons_minutes) != tuple(config.horizons_minutes):
        raise ValueError("Training dataset horizons do not match the experiment config.")
    if model_variant not in MULTIHORIZON_MODEL_VARIANTS:
        raise ValueError(f"Not a multi-horizon model variant: {model_variant}")
    if model_variant == "multi_horizon" and len(
        {dataset.patient_id for dataset in train_datasets}
    ) < 2:
        raise ValueError("Subject-invariant multi-horizon training needs two patients.")

    seed_everything(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ConcatDataset(train_datasets)
    loader = _make_train_loader(
        training,
        config.batch_size,
        samples_per_epoch=samples_per_epoch,
    )
    validation_loader = _make_eval_loader(validation_dataset, config.batch_size)
    model = build_model(
        n_channels=len(CHANNELS),
        n_subjects=len(train_datasets),
        sampling_rate=config.sampling_rate,
        variant=model_variant,
        horizons_minutes=config.horizons_minutes,
    ).to(device)
    horizon_counts = np.sum(
        [dataset.horizon_class_counts for dataset in train_datasets], axis=0
    )
    horizon_weights = []
    for counts in horizon_counts:
        present = counts > 0
        weights = np.ones(2, dtype=np.float32)
        if np.any(present):
            weights[present] = counts.sum() / (present.sum() * counts[present])
            weights /= weights[present].mean()
        horizon_weights.append(torch.as_tensor(weights, device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = epoch / max(config.epochs - 1, 1)
        reversal_strength = 2 / (1 + math.exp(-10 * progress)) - 1
        for signals, states, targets, subjects, _, _ in loader:
            signals = signals.to(device)
            states = states.to(device)
            targets = targets.to(device)
            subjects = subjects.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signals, reversal_strength=reversal_strength)
            state_logits = output["state_logits"]
            horizon_logits = output["horizon_logits"]
            if not isinstance(state_logits, torch.Tensor) or not isinstance(
                horizon_logits, torch.Tensor
            ):
                raise TypeError("Multi-horizon model did not return required logits.")

            state_loss = nn.functional.cross_entropy(state_logits, states)
            horizon_losses = []
            for horizon_index in range(len(config.horizons_minutes)):
                valid = targets[:, horizon_index] >= 0
                if torch.any(valid):
                    horizon_losses.append(
                        nn.functional.cross_entropy(
                            horizon_logits[valid, horizon_index],
                            targets[valid, horizon_index],
                            weight=horizon_weights[horizon_index],
                        )
                    )
            if not horizon_losses:
                raise ValueError("A training batch contains no valid horizon labels.")
            loss = state_loss + torch.stack(horizon_losses).mean()
            subject_logits = output["subject_logits"]
            if isinstance(subject_logits, torch.Tensor):
                loss = loss + config.subject_loss_weight * nn.functional.cross_entropy(
                    subject_logits, subjects
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach().item()) * len(states)
            seen += len(states)

        validation = predict_multihorizon_loader(model, validation_loader, device)
        horizon_ap = []
        for index in range(len(config.horizons_minutes)):
            valid = validation["horizon_labels"][:, index] >= 0
            labels = validation["horizon_labels"][valid, index]
            scores = validation["horizon_probabilities"][valid, index]
            if len(labels) and np.unique(labels).size == 2:
                horizon_ap.append(float(average_precision_score(labels, scores)))
        macro_state_f1 = float(
            f1_score(
                validation["state_labels"],
                validation["state_predictions"],
                average="macro",
                zero_division=0,
            )
        )
        selection_score = float(np.mean(horizon_ap)) if horizon_ap else macro_state_f1
        history.append(
            {
                "epoch": float(epoch + 1),
                "training_loss": running_loss / max(seen, 1),
                "validation_mean_horizon_pr_auc": selection_score,
                "validation_state_macro_f1": macro_state_f1,
            }
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_variant": model_variant,
                    "n_channels": len(CHANNELS),
                    "n_subjects": len(train_datasets),
                    "config": {
                        **config.to_dict(),
                        "model_variant": model_variant,
                    },
                    "train_subjects": [dataset.patient_id for dataset in train_datasets],
                    "validation_subject": validation_dataset.patient_id,
                    "best_epoch": best_epoch,
                    "best_validation_mean_horizon_pr_auc": best_score,
                    "training_samples_per_epoch": samples_per_epoch,
                    "checkpoint_protocol": (
                        "patient-separated validation; threshold calibration is "
                        "performed separately under false-alarm budgets"
                    ),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break

    return {
        "best_epoch": best_epoch,
        "best_validation_mean_horizon_pr_auc": best_score,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }


def train_final_multihorizon_model(
    train_datasets: list[MultiHorizonEEGDataset],
    config: ExperimentConfig,
    model_variant: str,
    checkpoint_path: Path,
    samples_per_epoch: int | None = 6000,
) -> dict[str, object]:
    """Fit a final deployment checkpoint with balanced cohort sampling."""
    if not train_datasets:
        raise ValueError("At least one training patient is required.")
    if len({dataset.patient_id for dataset in train_datasets}) != len(train_datasets):
        raise ValueError("Final training requires unique patient datasets.")
    if any(
        tuple(dataset.horizons_minutes) != tuple(config.horizons_minutes)
        for dataset in train_datasets
    ):
        raise ValueError("Training dataset horizons do not match the experiment config.")
    if model_variant not in MULTIHORIZON_MODEL_VARIANTS:
        raise ValueError(f"Not a multi-horizon model variant: {model_variant}")
    if model_variant == "multi_horizon" and len(train_datasets) < 2:
        raise ValueError("Subject-invariant multi-horizon training needs two patients.")
    if config.epochs < 1:
        raise ValueError("Final training requires at least one epoch.")

    seed_everything(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ConcatDataset(train_datasets)
    loader = _make_train_loader(
        training,
        config.batch_size,
        samples_per_epoch=samples_per_epoch,
        replacement=samples_per_epoch is not None,
    )
    model = build_model(
        n_channels=len(CHANNELS),
        n_subjects=len(train_datasets),
        sampling_rate=config.sampling_rate,
        variant=model_variant,
        horizons_minutes=config.horizons_minutes,
    ).to(device)
    horizon_counts = np.sum(
        [dataset.horizon_class_counts for dataset in train_datasets], axis=0
    )
    horizon_weights = []
    for counts in horizon_counts:
        present = counts > 0
        weights = np.ones(2, dtype=np.float32)
        if np.any(present):
            weights[present] = counts.sum() / (present.sum() * counts[present])
            weights /= weights[present].mean()
        horizon_weights.append(torch.as_tensor(weights, device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = epoch / max(config.epochs - 1, 1)
        reversal_strength = 2 / (1 + math.exp(-10 * progress)) - 1
        for signals, states, targets, subjects, _, _ in loader:
            signals = signals.to(device)
            states = states.to(device)
            targets = targets.to(device)
            subjects = subjects.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signals, reversal_strength=reversal_strength)
            state_logits = output["state_logits"]
            horizon_logits = output["horizon_logits"]
            if not isinstance(state_logits, torch.Tensor) or not isinstance(
                horizon_logits, torch.Tensor
            ):
                raise TypeError("Multi-horizon model did not return required logits.")
            state_loss = nn.functional.cross_entropy(state_logits, states)
            horizon_losses = []
            for horizon_index in range(len(config.horizons_minutes)):
                valid = targets[:, horizon_index] >= 0
                if torch.any(valid):
                    horizon_losses.append(
                        nn.functional.cross_entropy(
                            horizon_logits[valid, horizon_index],
                            targets[valid, horizon_index],
                            weight=horizon_weights[horizon_index],
                        )
                    )
            if not horizon_losses:
                raise ValueError("A training batch contains no valid horizon labels.")
            loss = state_loss + torch.stack(horizon_losses).mean()
            subject_logits = output["subject_logits"]
            if isinstance(subject_logits, torch.Tensor):
                loss = loss + config.subject_loss_weight * nn.functional.cross_entropy(
                    subject_logits, subjects
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach().item()) * len(states)
            seen += len(states)

        epoch_loss = running_loss / max(seen, 1)
        history.append({"epoch": float(epoch + 1), "training_loss": epoch_loss})
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "model_variant": model_variant,
            "n_channels": len(CHANNELS),
            "n_subjects": len(train_datasets),
            "config": {
                **config.to_dict(),
                "model_variant": model_variant,
            },
            "train_subjects": [dataset.patient_id for dataset in train_datasets],
            "validation_subject": None,
            "best_epoch": epoch + 1,
            "best_validation_mean_horizon_pr_auc": None,
            "training_samples_per_epoch": seen,
            "training_recordings": sum(
                len(dataset.recording_names) for dataset in train_datasets
            ),
            "training_windows": len(training),
            "training_history": history,
            "checkpoint_protocol": (
                "Final post-LOSO fit on all available patients using patient/state "
                "balanced sampling. This checkpoint is not an independent "
                "evaluation result."
            ),
        }
        torch.save(checkpoint_payload, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
        print(
            f"Final fit epoch {epoch + 1}/{config.epochs}: "
            f"loss={epoch_loss:.4f} windows={seen:,}"
        )

    return {
        "epochs": config.epochs,
        "training_recordings": checkpoint_payload["training_recordings"],
        "training_windows": len(training),
        "training_samples_per_epoch": checkpoint_payload["training_samples_per_epoch"],
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }


def train_model(
    train_datasets: list[HorizonEEGDataset],
    validation_dataset: HorizonEEGDataset,
    config: ExperimentConfig,
    model_variant: str,
    checkpoint_path: Path,
    device: torch.device | None = None,
    samples_per_epoch: int | None = None,
) -> dict[str, object]:
    if not train_datasets:
        raise ValueError("At least one training patient is required.")
    if len({dataset.patient_id for dataset in train_datasets}) < 2 and model_variant == "subject_invariant":
        raise ValueError("Subject-invariant training requires at least two training patients.")
    seed_everything(config.random_seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = ConcatDataset(train_datasets)
    loader = _make_train_loader(
        training,
        config.batch_size,
        samples_per_epoch=samples_per_epoch,
    )
    validation_loader = _make_eval_loader(validation_dataset, config.batch_size)
    model = build_model(
        n_channels=len(CHANNELS),
        n_subjects=len(train_datasets),
        sampling_rate=config.sampling_rate,
        variant=model_variant,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    prediction_criterion = nn.CrossEntropyLoss()
    subject_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_selection_f1 = -math.inf
    best_prediction_f1 = float("nan")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = epoch / max(config.epochs - 1, 1)
        reversal_strength = 2 / (1 + math.exp(-10 * progress)) - 1
        for signals, states, subjects in loader:
            signals = signals.to(device)
            states = states.to(device)
            subjects = subjects.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(signals, reversal_strength=reversal_strength)
            state_logits = output["state_logits"]
            prediction_logits = output["prediction_logits"]
            if not isinstance(state_logits, torch.Tensor) or not isinstance(
                prediction_logits, torch.Tensor
            ):
                raise TypeError("The model did not return state logits.")
            loss = criterion(state_logits, states)
            binary_mask = states < 2
            if torch.any(binary_mask):
                binary_targets = (states[binary_mask] == 1).long()
                loss = loss + prediction_criterion(
                    prediction_logits[binary_mask], binary_targets
                )
            subject_logits = output["subject_logits"]
            if isinstance(subject_logits, torch.Tensor):
                loss = loss + config.subject_loss_weight * subject_criterion(
                    subject_logits, subjects
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach().item()) * len(states)
            seen += len(states)

        y_true, y_pred, _, prediction_probs = predict_loader(model, validation_loader, device)
        state_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        binary_mask = (y_true == 0) | (y_true == 1)
        binary_true = (y_true[binary_mask] == 1).astype(np.int64)
        binary_pred = prediction_probs[binary_mask].argmax(axis=1)
        prediction_f1 = (
            float(f1_score(binary_true, binary_pred, zero_division=0))
            if len(binary_true) and np.unique(binary_true).size == 2
            else float("nan")
        )
        selection_f1 = prediction_f1 if np.isfinite(prediction_f1) else state_f1
        epoch_row = {
            "epoch": float(epoch + 1),
            "training_loss": running_loss / max(seen, 1),
            "validation_prediction_f1": prediction_f1,
            "validation_state_macro_f1": state_f1,
            "validation_selection_f1": selection_f1,
        }
        history.append(epoch_row)
        if selection_f1 > best_selection_f1:
            best_selection_f1 = selection_f1
            best_prediction_f1 = prediction_f1
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_variant": model_variant,
                    "n_channels": len(CHANNELS),
                    "n_subjects": len(train_datasets),
                    "config": config.to_dict(),
                    "train_subjects": [dataset.patient_id for dataset in train_datasets],
                    "validation_subject": validation_dataset.patient_id,
                    "best_epoch": best_epoch,
                    "best_validation_prediction_f1": best_prediction_f1,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= 4:
                break

    return {
        "best_epoch": best_epoch,
        "best_validation_prediction_f1": best_prediction_f1,
        "best_validation_selection_f1": best_selection_f1,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
    }


def load_checkpoint(
    checkpoint_path: Path, device: torch.device | None = None
) -> tuple[nn.Module, dict[str, object], torch.device]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ExperimentConfig.from_dict(payload["config"])
    model = build_model(
        n_channels=int(payload["n_channels"]),
        n_subjects=int(payload["n_subjects"]),
        sampling_rate=config.sampling_rate,
        variant=str(payload["model_variant"]),
        horizons_minutes=config.horizons_minutes,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload, device
