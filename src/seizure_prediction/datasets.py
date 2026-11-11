from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HorizonEEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Read one patient's cached EDF windows lazily from .npy memory maps."""

    def __init__(
        self,
        patient_cache: Path,
        horizon_minutes: int,
        subject_index: int = 0,
    ) -> None:
        manifest_path = patient_cache / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepared cache not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.patient_id = str(manifest["patient_id"])
        self.subject_index = subject_index
        self._records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._valid_indices: list[np.ndarray] = []
        self._cumulative: list[int] = []
        self.class_counts = np.zeros(3, dtype=np.int64)
        valid_labels: list[np.ndarray] = []
        total = 0
        for record in manifest["records"]:
            key = str(horizon_minutes)
            if key not in record["y_by_horizon"]:
                raise ValueError(
                    f"Cache {patient_cache} does not include horizon {horizon_minutes}."
                )
            x = np.load(patient_cache / record["x"], mmap_mode="r")
            y = np.load(patient_cache / record["y_by_horizon"][key], mmap_mode="r")
            starts = np.load(patient_cache / record["t"], mmap_mode="r")
            if not (len(x) == len(y) == len(starts)):
                raise ValueError(f"Cache arrays have different lengths in {record['recording']}.")
            valid_indices = np.flatnonzero(y >= 0)
            record_labels = np.asarray(y[valid_indices], dtype=np.int64)
            self.class_counts += np.bincount(record_labels, minlength=3)
            valid_labels.append(record_labels)
            self._records.append((x, y, starts))
            self._valid_indices.append(valid_indices)
            total += len(valid_indices)
            self._cumulative.append(total)
        if total == 0:
            raise ValueError(
                f"No labeled EEG windows for {self.patient_id} at {horizon_minutes} minutes."
            )
        self.labels = np.concatenate(valid_labels)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = int(np.searchsorted(self._cumulative, index, side="right"))
        previous_total = self._cumulative[record_index - 1] if record_index else 0
        local_index = int(self._valid_indices[record_index][index - previous_total])
        x, y, _ = self._records[record_index]
        signal = np.array(x[local_index], dtype=np.float32, copy=True)
        state = int(y[local_index])
        return (
            torch.from_numpy(signal),
            torch.tensor(state, dtype=torch.long),
            torch.tensor(self.subject_index, dtype=torch.long),
        )


class MultiHorizonEEGDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """Patient cache with one binary target and mask per forecast horizon."""

    def __init__(
        self,
        patient_cache: Path,
        horizons_minutes: tuple[int, ...],
        subject_index: int = 0,
    ) -> None:
        manifest_path = patient_cache / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepared cache not found: {manifest_path}")
        if not horizons_minutes or len(set(horizons_minutes)) != len(horizons_minutes):
            raise ValueError("Multi-horizon dataset requires unique prepared horizons.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.patient_id = str(manifest["patient_id"])
        self.subject_index = subject_index
        self.horizons_minutes = horizons_minutes
        self.recording_names: list[str] = []
        self.recording_offsets_seconds: dict[str, float] = {
            name.lower(): float(value)
            for name, value in manifest.get("recording_offsets_seconds", {}).items()
        }
        self.seizures_by_recording: dict[str, list[tuple[float, float]]] = {
            name.lower(): [(float(onset), float(offset)) for onset, offset in events]
            for name, events in manifest.get("seizures_by_recording", {}).items()
        }
        self._records: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._valid_indices: list[np.ndarray] = []
        self._cumulative: list[int] = []
        states_by_record: list[np.ndarray] = []
        targets_by_record: list[np.ndarray] = []
        record_ids: list[np.ndarray] = []
        times: list[np.ndarray] = []
        class_counts = np.zeros(3, dtype=np.int64)
        horizon_class_counts = np.zeros((len(horizons_minutes), 2), dtype=np.int64)
        total = 0

        for record_index, record in enumerate(manifest["records"]):
            labels = []
            for horizon in horizons_minutes:
                key = str(horizon)
                if key not in record["y_by_horizon"]:
                    raise ValueError(
                        f"Cache {patient_cache} does not include horizon {horizon}; "
                        "run the `relabel` command first."
                    )
                labels.append(
                    np.load(
                        patient_cache / record["y_by_horizon"][key],
                        mmap_mode="r",
                    )
                )
            state_matrix = np.stack([np.asarray(values) for values in labels], axis=1)
            x = np.load(patient_cache / record["x"], mmap_mode="r")
            starts = np.load(patient_cache / record["t"], mmap_mode="r")
            if not (len(x) == len(starts) == state_matrix.shape[0]):
                raise ValueError(
                    f"Cache arrays have different lengths in {record['recording']}."
                )
            valid_indices = np.flatnonzero(np.any(state_matrix >= 0, axis=1))
            if not len(valid_indices):
                continue

            state_targets = np.full(len(valid_indices), -1, dtype=np.int64)
            selected = state_matrix[valid_indices]
            for horizon_index in range(len(horizons_minutes)):
                available = selected[:, horizon_index] >= 0
                state_targets[available] = selected[available, horizon_index]
            horizon_targets = np.where(
                (selected >= 0) & (selected < 2), selected, -1
            ).astype(np.int64)
            for horizon_index in range(len(horizons_minutes)):
                valid_targets = horizon_targets[:, horizon_index]
                horizon_class_counts[horizon_index] += np.bincount(
                    valid_targets[valid_targets >= 0], minlength=2
                )
            class_counts += np.bincount(
                state_targets[state_targets >= 0], minlength=3
            )
            self.recording_names.append(str(record["recording"]))
            self._records.append((x, state_matrix, starts))
            self._valid_indices.append(valid_indices)
            states_by_record.append(state_targets)
            targets_by_record.append(horizon_targets)
            record_ids.append(
                np.full(len(valid_indices), len(self.recording_names) - 1, dtype=np.int64)
            )
            times.append(np.asarray(starts[valid_indices], dtype=np.float32))
            total += len(valid_indices)
            self._cumulative.append(total)

        if total == 0:
            raise ValueError(f"No labeled EEG windows for {self.patient_id}.")
        self.labels = np.concatenate(states_by_record)
        self.horizon_labels = np.concatenate(targets_by_record)
        self.recording_indices = np.concatenate(record_ids)
        self.starts_seconds = np.concatenate(times)
        self.class_counts = class_counts
        self.horizon_class_counts = horizon_class_counts

    def __len__(self) -> int:
        return self._cumulative[-1]

    def window(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = int(np.searchsorted(self._cumulative, index, side="right"))
        previous_total = self._cumulative[record_index - 1] if record_index else 0
        local_index = int(self._valid_indices[record_index][index - previous_total])
        x, _, _ = self._records[record_index]
        return np.array(x[local_index], dtype=np.float32, copy=True)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = int(np.searchsorted(self._cumulative, index, side="right"))
        previous_total = self._cumulative[record_index - 1] if record_index else 0
        local_index = int(self._valid_indices[record_index][index - previous_total])
        x, states, _ = self._records[record_index]
        state_row = np.asarray(states[local_index], dtype=np.int64)
        available = state_row[state_row >= 0]
        state = int(available[-1])
        targets = np.where((state_row >= 0) & (state_row < 2), state_row, -1)
        return (
            torch.from_numpy(np.array(x[local_index], dtype=np.float32, copy=True)),
            torch.tensor(state, dtype=torch.long),
            torch.as_tensor(targets, dtype=torch.long),
            torch.tensor(self.subject_index, dtype=torch.long),
            torch.tensor(record_index, dtype=torch.long),
            torch.tensor(float(self._records[record_index][2][local_index])),
        )
