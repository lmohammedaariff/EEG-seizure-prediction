from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from seizure_prediction.training import _make_train_loader


class _PatientDataset(TensorDataset):
    def __init__(self, patient_id: str) -> None:
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        super().__init__(
            torch.zeros((len(labels), 1)),
            torch.from_numpy(labels),
            torch.zeros(len(labels), dtype=torch.long),
        )
        self.patient_id = patient_id
        self.labels = labels
        self.class_counts = np.bincount(labels, minlength=3)


def test_train_loader_caps_samples_per_epoch() -> None:
    dataset = ConcatDataset([_PatientDataset("chb01"), _PatientDataset("chb02")])

    loader = _make_train_loader(dataset, batch_size=2, samples_per_epoch=5)

    assert loader.sampler.num_samples == 5


def test_train_loader_defaults_to_full_weighted_epoch() -> None:
    dataset = ConcatDataset([_PatientDataset("chb01"), _PatientDataset("chb02")])

    loader = _make_train_loader(dataset, batch_size=2)

    assert loader.sampler.num_samples == len(dataset)


def test_train_loader_rejects_nonpositive_sample_cap() -> None:
    dataset = ConcatDataset([_PatientDataset("chb01"), _PatientDataset("chb02")])

    with pytest.raises(ValueError, match="Samples per epoch must be positive"):
        _make_train_loader(dataset, batch_size=2, samples_per_epoch=0)
