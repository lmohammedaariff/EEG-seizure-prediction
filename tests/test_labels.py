from pathlib import Path

import numpy as np

from seizure_prediction.edf import (
    RecordingAnnotations,
    parse_chbmit_summary,
    recording_offsets,
)
from seizure_prediction.labels import (
    EXCLUDED,
    ICTAL,
    INTERICTAL,
    PREICTAL,
    Seizure,
    label_windows,
    prediction_labels,
)


def test_parse_chbmit_summary_groups_seizures_by_recording(tmp_path: Path) -> None:
    summary = tmp_path / "chb01-summary.txt"
    summary.write_text(
        "File Name: chb01_01.edf\n"
        "File Start Time: 23:50:00\n"
        "Number of Seizures in File: 0\n"
        "File Name: chb01_02.edf\n"
        "File Start Time: 00:10:00\n"
        "Number of Seizures in File: 2\n"
        "Seizure 1 Start Time: 100 seconds\n"
        "Seizure 1 End Time: 120 seconds\n"
        "Seizure 2 Start Time: 500 seconds\n"
        "Seizure 2 End Time: 540 seconds\n",
        encoding="utf-8",
    )
    annotations = parse_chbmit_summary(summary).seizures_by_file
    assert annotations["chb01_01.edf"] == []
    assert annotations["chb01_02.edf"] == [
        Seizure(100.0, 120.0),
        Seizure(500.0, 540.0),
    ]


def test_recording_offsets_roll_over_midnight() -> None:
    files = [Path("chb01_01.edf"), Path("chb01_02.edf")]
    annotations = RecordingAnnotations(
        seizures_by_file={},
        start_times_seconds={
            "chb01_01.edf": 23 * 3600 + 50 * 60,
            "chb01_02.edf": 10 * 60,
        },
    )
    offsets = recording_offsets(
        files,
        annotations,
        {"chb01_01.edf": 20 * 60, "chb01_02.edf": 20 * 60},
    )
    assert offsets["chb01_02.edf"] == 24 * 3600 + 10 * 60
    event = Seizure(offsets["chb01_01.edf"] + 1195, offsets["chb01_01.edf"] + 1205)
    next_window = np.array([offsets["chb01_02.edf"]])
    assert label_windows(
        next_window,
        window_seconds=10,
        seizures=[event],
        preictal_minutes=5,
        sph_minutes=0,
        interictal_buffer_minutes=60,
    ).tolist() == [ICTAL]


def test_recording_offsets_use_header_timestamps_not_filename_order() -> None:
    files = [Path("chb02_16+.edf"), Path("chb02_16.edf")]
    annotations = RecordingAnnotations({}, {})
    offsets = recording_offsets(
        files,
        annotations,
        {"chb02_16+.edf": 3600, "chb02_16.edf": 3600},
        {"chb02_16+.edf": 4717.0, "chb02_16.edf": 0.0},
    )

    assert offsets == {"chb02_16+.edf": 4717.0, "chb02_16.edf": 0.0}


def test_state_labels_respect_sop_sph_and_ambiguous_buffer() -> None:
    starts = np.array([0, 2500, 2600, 2700, 3000, 3290, 3300, 3600, 3750])
    labels = label_windows(
        starts,
        window_seconds=10,
        seizures=[Seizure(3600, 3630)],
        preictal_minutes=10,
        sph_minutes=5,
        interictal_buffer_minutes=2,
    )
    assert labels.tolist() == [
        INTERICTAL,
        INTERICTAL,
        EXCLUDED,
        PREICTAL,
        PREICTAL,
        PREICTAL,
        EXCLUDED,
        ICTAL,
        INTERICTAL,
    ]


def test_prediction_task_excludes_ictal_and_ambiguous_labels() -> None:
    labels = prediction_labels(
        np.array([INTERICTAL, PREICTAL, ICTAL, EXCLUDED, PREICTAL])
    )
    assert labels.tolist() == [0, 1, 1]
