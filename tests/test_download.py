from pathlib import Path

import pytest

from scripts.download_chbmit import _safe_relative_path, _select_complete_patients


def test_select_complete_patients_preserves_reserve_and_maximizes_subject_count(
    tmp_path: Path,
) -> None:
    mib = 1024 * 1024
    patient_sizes = {
        f"chb{index:02d}": {f"chb{index:02d}/record.edf": 4 * mib}
        for index in range(1, 6)
    }
    selected, costs = _select_complete_patients(
        patient_sizes,
        target=tmp_path,
        free_bytes=20 * mib,
        reserve_bytes=11 * mib,
    )
    assert len(selected) == 2
    assert sum(costs[patient] for patient in selected) <= 9 * mib


def test_select_complete_patients_accounts_for_resumable_partial_files(
    tmp_path: Path,
) -> None:
    mib = 1024 * 1024
    patient_sizes = {
        "chb01": {"chb01/record.edf": 5 * mib},
        "chb02": {"chb02/record.edf": 5 * mib},
    }
    partial = tmp_path / "chb01" / "record.edf.part"
    partial.parent.mkdir()
    partial.write_bytes(b"x" * (4 * mib))
    selected, costs = _select_complete_patients(
        patient_sizes,
        target=tmp_path,
        free_bytes=6 * mib,
        reserve_bytes=2 * mib,
    )
    assert selected == ["chb01"]
    assert costs["chb01"] == mib


def test_safe_path_decodes_physionet_url_escaped_plus_filename() -> None:
    assert _safe_relative_path("chb02/chb02_16%2B.edf").as_posix() == (
        "chb02/chb02_16+.edf"
    )


def test_safe_path_rejects_encoded_parent_directory() -> None:
    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_relative_path("chb02/%2e%2e/outside.edf")
