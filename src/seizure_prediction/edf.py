from __future__ import annotations

from datetime import timezone
from dataclasses import dataclass
import json
import re
from pathlib import Path

import mne
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import CHANNELS, ExperimentConfig
from .labels import Seizure, label_windows

_FILE_RE = re.compile(r"^\s*File Name:\s*(.+?)\s*$", re.IGNORECASE)
_FILE_START_RE = re.compile(
    r"^\s*File Start Time:\s*(\d{1,2}):(\d{2}):(\d{2})\s*$",
    re.IGNORECASE,
)
_SEIZURE_RE = re.compile(
    r"^\s*Seizure(?:\s+\d+)?\s+(Start|End)\s+Time:\s*([\d.]+)\s*seconds?",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"^\s*(\d{1,2})\.(\d{2})\.(\d{2})\s*$")


@dataclass(frozen=True)
class RecordingAnnotations:
    seizures_by_file: dict[str, list[Seizure]]
    start_times_seconds: dict[str, float | None]


def parse_siena_annotations(path: Path) -> dict[str, list[Seizure]]:
    """Parse PhysioNet Siena Seizures-list files into per-EDF relative times.

    Siena annotations use time-of-day clocks. Each seizure is converted to
    seconds after its EDF's registration start; modulo-day arithmetic handles
    recordings that cross midnight.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Siena seizure list not found: {path}")
    events: dict[str, list[Seizure]] = {}
    recording: str | None = None
    registration_start: int | None = None
    pending_start: int | None = None
    pending_end: int | None = None

    def clock_seconds(value: str, line: str) -> int:
        # Siena summary text contains occasional annotation typos such as
        # "1 6.49.25" and trailing editorial text. Join a split tens-hour
        # digit and parse the first clock, retaining the source line in errors.
        value = re.sub(r"(\d)\s+(\d)(?=[.:]\d{2}[.:]\d{2})", r"\1\2", value)
        match = re.search(r"(?<!\d)(\d{1,2})[.:](\d{2})[.:](\d{2})", value)
        if not match:
            raise ValueError(f"Invalid Siena time-of-day in {path}: {line.strip()}")
        hours, minutes, seconds = map(int, match.groups())
        if hours > 23 or minutes > 59 or seconds > 59:
            raise ValueError(f"Invalid Siena time-of-day in {path}: {line.strip()}")
        return hours * 3600 + minutes * 60 + seconds

    def flush() -> None:
        nonlocal pending_start, pending_end
        if pending_start is not None or pending_end is not None:
            if recording is None or registration_start is None:
                raise ValueError(f"Incomplete Siena seizure metadata in {path}.")
            if pending_start is None or pending_end is None:
                raise ValueError(f"Incomplete Siena seizure interval in {path}.")
            onset = float((pending_start - registration_start) % 86400)
            offset = float((pending_end - registration_start) % 86400)
            if offset <= onset:
                offset += 86400
            events.setdefault(recording.lower(), []).append(Seizure(onset, offset))
        pending_start = None
        pending_end = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        lower = line.strip().lower()
        if lower.startswith("file name:"):
            flush()
            next_recording = Path(line.split(":", 1)[1].strip()).name
            # Correct the known PNO6 transcription typo in the PhysioNet file.
            next_recording = re.sub(r"(?i)(?<=pn)o(?=\d)", "0", next_recording)
            if recording is None or next_recording.lower() != recording.lower():
                registration_start = None
            recording = next_recording
            continue
        if lower.startswith("registration start time:"):
            value = line.split(":", 1)[1].strip()
            registration_start = clock_seconds(value, line)
            continue
        if lower.startswith("seizure n"):
            flush()
            continue
        if lower.startswith("seizure start time:"):
            pending_start = clock_seconds(line.split(":", 1)[1].strip(), line)
        elif lower.startswith("seizure end time:"):
            pending_end = clock_seconds(line.split(":", 1)[1].strip(), line)
        elif lower.startswith("start time:"):
            pending_start = clock_seconds(line.split(":", 1)[1].strip(), line)
        elif lower.startswith("end time:"):
            pending_end = clock_seconds(line.split(":", 1)[1].strip(), line)
    flush()
    return events


def parse_chbmit_summary(path: Path) -> RecordingAnnotations:
    """Read per-recording seizure seconds from a CHB-MIT summary text file."""
    if not path.is_file():
        raise FileNotFoundError(f"CHB-MIT summary file not found: {path}")
    mapping: dict[str, list[Seizure]] = {}
    start_times: dict[str, float | None] = {}
    current_file: str | None = None
    current_start_time: float | None = None
    pending: dict[int, dict[str, float]] = {}

    def flush() -> None:
        if current_file is None:
            return
        intervals = []
        for seizure_number, bounds in sorted(pending.items()):
            if "start" not in bounds or "end" not in bounds:
                raise ValueError(
                    f"Incomplete seizure annotation {seizure_number} in {current_file}."
                )
            intervals.append(Seizure(bounds["start"], bounds["end"]))
        mapping[Path(current_file).name.lower()] = intervals
        start_times[Path(current_file).name.lower()] = current_start_time

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        file_match = _FILE_RE.match(line)
        if file_match:
            flush()
            current_file = file_match.group(1).strip()
            current_start_time = None
            pending = {}
            continue
        if current_file is None:
            continue
        start_match = _FILE_START_RE.match(line)
        if start_match:
            hours, minutes, seconds = (int(value) for value in start_match.groups())
            if hours > 23 or minutes > 59 or seconds > 59:
                raise ValueError(f"Invalid file start time in {path}: {line.strip()}")
            current_start_time = hours * 3600 + minutes * 60 + seconds
            continue
        seizure_match = _SEIZURE_RE.match(line)
        if not seizure_match:
            continue
        number_match = re.search(r"Seizure\s+(\d+)", line, re.IGNORECASE)
        number = int(number_match.group(1)) if number_match else 1
        kind = seizure_match.group(1).lower()
        pending.setdefault(number, {})[kind] = float(seizure_match.group(2))
    flush()
    return RecordingAnnotations(mapping, start_times)


def _normalize_channel_name(name: str) -> str:
    normalized = re.sub(r"\bEEG\b", "", name.upper())
    normalized = re.sub(r"-(?:REF|LE|AVG|AR)$", "", normalized)
    normalized = re.sub(r"\s+", "", normalized).replace("_", "-")
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    for legacy, modern in aliases.items():
        normalized = re.sub(rf"\b{legacy}\b", modern, normalized)
    return normalized


def standardize_channels(
    data_volts: np.ndarray, channel_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Select a fixed bipolar montage and zero-fill missing channels."""
    source_by_name: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        source_by_name.setdefault(_normalize_channel_name(name), index)
    result = np.zeros((len(CHANNELS), data_volts.shape[1]), dtype=np.float32)
    matched: list[str] = []
    for target_index, canonical_name in enumerate(CHANNELS):
        source_index = source_by_name.get(_normalize_channel_name(canonical_name))
        if source_index is not None:
            result[target_index] = data_volts[source_index].astype(np.float32) * 1e6
            matched.append(canonical_name)
            continue

        # Siena EDFs contain referential 10-20 electrodes rather than the
        # bipolar derivations used by CHB-MIT. Build the same bipolar montage
        # from electrode pairs when a direct derivation is not present.
        first, second = canonical_name.split("-", maxsplit=1)
        first_index = source_by_name.get(_normalize_channel_name(first))
        second_index = source_by_name.get(_normalize_channel_name(second))
        if first_index is None or second_index is None:
            continue
        result[target_index] = (
            data_volts[first_index].astype(np.float32)
            - data_volts[second_index].astype(np.float32)
        ) * 1e6
        matched.append(canonical_name)
    if len(matched) < 4:
        raise ValueError(
            "Fewer than four channels matched the supported CHB-MIT bipolar montage. "
            f"Found {matched}; source channels were {channel_names}."
        )
    return result, matched


def _resample_float32_chunked(
    data: np.ndarray,
    source_rate: int,
    target_rate: int,
    chunk_seconds: int = 1800,
    context_seconds: int = 10,
) -> np.ndarray:
    """Fourier-resample long float32 EEG in overlapped chunks.

    MNE's Fourier resampler requires float64 and allocates multiple full-length
    work arrays. Chunking preserves that resampling method while bounding its
    temporary memory on long Siena recordings.
    """
    if data.ndim != 2 or data.dtype != np.float32:
        raise ValueError("Chunked resampling expects float32 [channels, samples].")
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be positive.")
    chunk_samples = source_rate * chunk_seconds
    context_samples = source_rate * context_seconds
    ratio = target_rate / source_rate
    target_length = int(round(data.shape[-1] * ratio))
    result = np.empty((data.shape[0], target_length), dtype=np.float32)
    for start in range(0, data.shape[-1], chunk_samples):
        stop = min(start + chunk_samples, data.shape[-1])
        left = max(0, start - context_samples)
        right = min(data.shape[-1], stop + context_samples)
        chunk = mne.filter.resample(
            data[:, left:right].astype(np.float64),
            up=target_rate,
            down=source_rate,
            npad="auto",
            verbose="ERROR",
        )
        target_start = int(round(start * ratio))
        target_stop = min(target_length, int(round(stop * ratio)))
        chunk_start = int(round((start - left) * ratio))
        chunk_stop = chunk_start + target_stop - target_start
        if chunk_stop > chunk.shape[-1]:
            raise RuntimeError("Resampled chunk is shorter than its requested output range.")
        result[:, target_start:target_stop] = chunk[:, chunk_start:chunk_stop]
    return result


def preprocess_edf(path: Path, config: ExperimentConfig) -> tuple[np.ndarray, list[str]]:
    """Load, filter, resample, and standardize one EDF, returning microvolts."""
    if not path.is_file():
        raise FileNotFoundError(f"EDF file not found: {path}")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        source_rate = float(raw.info["sfreq"])
        if source_rate <= 2 * config.bandpass_low_hz:
            raise ValueError(f"Sampling rate is too low to filter: {source_rate} Hz")
        high_hz = min(config.bandpass_high_hz, source_rate / 2 - 1)
        if high_hz <= config.bandpass_low_hz:
            raise ValueError(
                f"Sampling rate is too low for the configured band: {source_rate} Hz"
            )

        eeg_names = [
            name
            for name, kind in zip(raw.ch_names, raw.get_channel_types())
            if kind == "eeg"
        ]
        if not eeg_names:
            raise ValueError(f"No EEG channels were found in {path}.")
        raw.pick(eeg_names)
        raw.load_data(verbose="ERROR")
        raw.filter(
            l_freq=config.bandpass_low_hz,
            h_freq=high_hz,
            picks="eeg",
            verbose="ERROR",
        )
        if config.notch_hz < source_rate / 2:
            raw.notch_filter(freqs=[config.notch_hz], picks="eeg", verbose="ERROR")

        # The preloaded Raw already owns this array. Reading through get_data()
        # makes another full-size float64 copy, which is prohibitive for long
        # Siena recordings.
        volts = raw._data
        data, matched = standardize_channels(volts, eeg_names)
        del volts
        raw.close()
        raw = None
        if int(round(source_rate)) != config.sampling_rate:
            data = _resample_float32_chunked(
                data,
                source_rate=int(round(source_rate)),
                target_rate=config.sampling_rate,
            )
    finally:
        if raw is not None:
            raw.close()
    return data, matched


def recording_metadata(path: Path) -> tuple[float, float | None]:
    """Read EDF duration and absolute start timestamp without loading its signal."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    try:
        measured_start = raw.info.get("meas_date")
        if measured_start is not None and measured_start.tzinfo is None:
            measured_start = measured_start.replace(tzinfo=timezone.utc)
        start_timestamp = (
            float(measured_start.timestamp()) if measured_start is not None else None
        )
        return float(raw.n_times / raw.info["sfreq"]), start_timestamp
    finally:
        raw.close()


def recording_duration(path: Path) -> float:
    """Read EDF duration from its header without preloading the EEG."""
    return recording_metadata(path)[0]


def recording_offsets(
    edfs: list[Path],
    annotations: RecordingAnnotations,
    durations: dict[str, float],
    start_timestamps: dict[str, float] | None = None,
) -> dict[str, float]:
    """Map recordings onto a timeline, preferring absolute EDF header timestamps."""
    if start_timestamps is not None and all(
        path.name.lower() in start_timestamps for path in edfs
    ):
        timestamps = {
            path.name.lower(): float(start_timestamps[path.name.lower()])
            for path in edfs
        }
        if not all(np.isfinite(value) for value in timestamps.values()):
            raise ValueError("EDF start timestamps must be finite.")
        origin = min(timestamps.values())
        return {
            name: timestamp - origin for name, timestamp in timestamps.items()
        }

    offsets: dict[str, float] = {}
    day_offset = 0.0
    previous_offset: float | None = None
    previous_duration = 0.0
    for path in edfs:
        key = path.name.lower()
        clock = annotations.start_times_seconds.get(key)
        if clock is None:
            offset = (
                previous_offset + previous_duration
                if previous_offset is not None
                else 0.0
            )
        else:
            offset = day_offset + clock
            while previous_offset is not None and offset < previous_offset:
                day_offset += 24 * 60 * 60
                offset = day_offset + clock
        offsets[key] = offset
        previous_offset = offset
        previous_duration = durations[key]
    return offsets


def window_recording(
    data: np.ndarray,
    seizures: list[Seizure],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create all overlapping fixed-duration windows, including excluded labels."""
    window = config.window_samples
    step = config.step_samples
    if data.ndim != 2 or data.shape[1] < window:
        raise ValueError("Recording must contain at least one complete EEG window.")
    views = sliding_window_view(data, window_shape=window, axis=1)
    windows = np.moveaxis(views[:, ::step, :], 1, 0).copy()
    means = windows.mean(axis=-1, keepdims=True)
    scales = windows.std(axis=-1, keepdims=True)
    windows = (windows - means) / np.maximum(scales, 1e-6)
    starts = np.arange(windows.shape[0], dtype=np.float64) * step / config.sampling_rate
    labels = label_windows(
        starts,
        config.window_seconds,
        seizures,
        config.preictal_minutes,
        config.sph_minutes,
        config.interictal_buffer_minutes,
        config.min_ictal_overlap_seconds,
    )
    return windows, labels, starts


def prepare_patient(
    patient_dir: Path,
    cache_dir: Path,
    config: ExperimentConfig,
    max_recordings: int | None = None,
) -> dict[str, object]:
    """Preprocess a patient's EDFs into cached windows and horizon-specific labels."""
    patient_id = patient_dir.name.lower()
    summaries = sorted(patient_dir.glob(f"{patient_id}-summary.txt"))
    if not summaries:
        raise FileNotFoundError(f"No {patient_id}-summary.txt found in {patient_dir}.")
    annotations = parse_chbmit_summary(summaries[0])
    edfs = sorted(patient_dir.glob("*.edf"))
    if max_recordings is not None:
        if max_recordings < 1:
            raise ValueError("max_recordings must be positive when provided.")
        edfs = edfs[:max_recordings]
    if not edfs:
        raise FileNotFoundError(f"No EDF recordings found in {patient_dir}.")
    available_recordings = {path.name.lower() for path in edfs}
    missing_summary_recordings = sorted(
        set(annotations.seizures_by_file) - available_recordings
    )
    if max_recordings is None and missing_summary_recordings:
        raise ValueError(
            f"{patient_id} is incomplete: its summary references EDF recordings "
            f"that are missing from {patient_dir}: {missing_summary_recordings}."
        )
    metadata = {
        path.name.lower(): recording_metadata(path) for path in edfs
    }
    durations = {name: item[0] for name, item in metadata.items()}
    start_timestamps = {
        name: item[1]
        for name, item in metadata.items()
        if item[1] is not None
    }
    offsets = recording_offsets(edfs, annotations, durations, start_timestamps)
    missing_seizure_recordings = sorted(
        file_name
        for file_name, events in annotations.seizures_by_file.items()
        if events and file_name not in offsets
    )
    if missing_seizure_recordings:
        raise ValueError(
            f"{patient_id} has seizure annotations without cached EDF recordings: "
            f"{missing_seizure_recordings}."
        )
    patient_seizures = [
        Seizure(offsets[file_name] + seizure.onset_seconds,
                offsets[file_name] + seizure.offset_seconds)
        for file_name, seizures in annotations.seizures_by_file.items()
        if file_name in offsets
        for seizure in seizures
    ]

    patient_cache = cache_dir / patient_id
    patient_cache.mkdir(parents=True, exist_ok=True)
    manifest_path = patient_cache / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    records: list[dict[str, object]] = []
    for recording_index, edf_path in enumerate(edfs, start=1):
        print(
            f"  {patient_id} recording {recording_index}/{len(edfs)}: "
            f"{edf_path.name}"
        )
        signal, matched = preprocess_edf(edf_path, config)
        windows, _, starts = window_recording(signal, [], config)
        absolute_starts = starts + offsets[edf_path.name.lower()]
        stem = edf_path.stem.lower()
        x_path = patient_cache / f"{stem}.x.npy"
        t_path = patient_cache / f"{stem}.t.npy"
        np.save(x_path, windows.astype(np.float32), allow_pickle=False)
        np.save(t_path, starts.astype(np.float32), allow_pickle=False)
        labels_by_horizon: dict[str, str] = {}
        counts_by_horizon: dict[str, list[int]] = {}
        for horizon in sorted(set(config.horizons_minutes)):
            labels = label_windows(
                absolute_starts,
                config.window_seconds,
                patient_seizures,
                horizon,
                config.sph_minutes,
                config.interictal_buffer_minutes,
                config.min_ictal_overlap_seconds,
            )
            y_path = patient_cache / f"{stem}.y-h{horizon}.npy"
            np.save(y_path, labels.astype(np.int8), allow_pickle=False)
            labels_by_horizon[str(horizon)] = y_path.name
            counts_by_horizon[str(horizon)] = np.bincount(
                labels[labels >= 0].astype(np.int64), minlength=3
            ).tolist()
        records.append(
            {
                "recording": edf_path.name,
                "x": x_path.name,
                "t": t_path.name,
                "y_by_horizon": labels_by_horizon,
                "windows": int(windows.shape[0]),
                "class_counts_by_horizon": counts_by_horizon,
                "matched_channels": matched,
            }
        )

    manifest = {
        "patient_id": patient_id,
        "config": config.to_dict(),
        "channels": list(CHANNELS),
        "recording_offsets_seconds": offsets,
        "seizures_by_recording": {
            name: [[event.onset_seconds, event.offset_seconds] for event in events]
            for name, events in annotations.seizures_by_file.items()
            if name in offsets
        },
        "records": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def prepare_dataset(
    data_dir: Path,
    cache_dir: Path,
    config: ExperimentConfig,
    max_recordings_per_patient: int | None = None,
) -> list[dict[str, object]]:
    patients = sorted(
        path
        for path in data_dir.glob("chb*")
        if path.is_dir()
        and any(path.glob("*.edf"))
        and not any(path.glob("*.edf.part"))
    )
    if not patients:
        raise FileNotFoundError(
            f"No chb* patient folders found in {data_dir}. "
            "Download and extract CHB-MIT first; see README.md."
        )
    manifests = []
    for index, patient in enumerate(patients, start=1):
        print(f"Preparing patient {index}/{len(patients)}: {patient.name}")
        manifests.append(
            prepare_patient(
                patient,
                cache_dir,
                config,
                max_recordings=max_recordings_per_patient,
            )
        )
    return manifests


def relabel_cache_horizons(
    data_dir: Path,
    cache_dir: Path,
    horizons_minutes: tuple[int, ...],
) -> list[dict[str, object]]:
    """Add horizon-specific labels to existing windows without reprocessing EDFs."""
    horizons = tuple(sorted(set(horizons_minutes)))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("Provide one or more positive horizons.")
    patients = sorted(
        path
        for path in cache_dir.glob("chb*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if not patients:
        raise FileNotFoundError(f"No prepared caches found under {cache_dir}.")

    manifests: list[dict[str, object]] = []
    shared_fields = (
        "sampling_rate",
        "window_seconds",
        "overlap",
        "sph_minutes",
        "interictal_buffer_minutes",
        "min_ictal_overlap_seconds",
    )
    reference_values: dict[str, object] | None = None

    for patient_cache in patients:
        patient_id = patient_cache.name.lower()
        patient_dir = data_dir / patient_id
        manifest_path = patient_cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_config = ExperimentConfig.from_dict(manifest["config"])
        current_values = current_config.to_dict()
        if reference_values is None:
            reference_values = current_values
        mismatched = [
            key
            for key in shared_fields
            if current_values[key] != reference_values[key]
        ]
        if mismatched:
            raise ValueError(
                f"Prepared caches disagree on {mismatched}; cannot relabel consistently."
            )

        summary_path = patient_dir / f"{patient_id}-summary.txt"
        annotations = parse_chbmit_summary(summary_path)
        offsets = {
            str(name).lower(): float(value)
            for name, value in manifest.get("recording_offsets_seconds", {}).items()
        }
        if not offsets:
            raise ValueError(f"Cache {patient_id} has no recording offsets.")
        expected_recordings = {
            str(record["recording"]).lower() for record in manifest["records"]
        }
        edfs = sorted(patient_dir.glob("*.edf"))
        if edfs:
            raw_recordings = {path.name.lower() for path in edfs}
            if raw_recordings != expected_recordings:
                raise ValueError(
                    f"Cache {patient_id} does not cover its raw EDF set; "
                    f"uncached={sorted(raw_recordings - expected_recordings)}, "
                    f"missing_raw={sorted(expected_recordings - raw_recordings)}."
                )
            metadata = {path.name.lower(): recording_metadata(path) for path in edfs}
            durations = {name: item[0] for name, item in metadata.items()}
            start_timestamps = {
                name: item[1]
                for name, item in metadata.items()
                if item[1] is not None
            }
            offsets = recording_offsets(
                edfs,
                annotations,
                durations,
                start_timestamps,
            )
        missing_summary_recordings = sorted(
            set(annotations.seizures_by_file) - offsets.keys()
        )
        if missing_summary_recordings:
            raise ValueError(
                f"Cache {patient_id} is incomplete: its summary references EDF "
                f"recordings without cached offsets: {missing_summary_recordings}."
            )
        missing_offsets = sorted(expected_recordings - offsets.keys())
        if missing_offsets:
            raise ValueError(
                f"Cache {patient_id} lacks offsets for recordings {missing_offsets}."
            )
        patient_events = [
            Seizure(
                offsets[file_name] + event.onset_seconds,
                offsets[file_name] + event.offset_seconds,
            )
            for file_name, events in annotations.seizures_by_file.items()
            if file_name in offsets
            for event in events
        ]

        for record in manifest["records"]:
            recording = str(record["recording"]).lower()
            starts = np.load(patient_cache / record["t"], mmap_mode="r")
            absolute_starts = np.asarray(starts, dtype=np.float64) + offsets[recording]
            labels_by_horizon: dict[str, str] = {}
            counts_by_horizon: dict[str, list[int]] = {}
            stem = Path(str(record["x"])).name.removesuffix(".x.npy")
            for horizon in horizons:
                labels = label_windows(
                    absolute_starts,
                    current_config.window_seconds,
                    patient_events,
                    horizon,
                    current_config.sph_minutes,
                    current_config.interictal_buffer_minutes,
                    current_config.min_ictal_overlap_seconds,
                )
                y_name = f"{stem}.y-h{horizon}.npy"
                np.save(patient_cache / y_name, labels.astype(np.int8), allow_pickle=False)
                labels_by_horizon[str(horizon)] = y_name
                counts_by_horizon[str(horizon)] = np.bincount(
                    labels[labels >= 0].astype(np.int64), minlength=3
                ).tolist()
            record["y_by_horizon"] = labels_by_horizon
            record["class_counts_by_horizon"] = counts_by_horizon

        manifest["config"] = ExperimentConfig.from_dict(
            {
                **current_values,
                "horizons_minutes": list(horizons),
                "preictal_minutes": max(horizons),
            }
        ).to_dict()
        manifest["recording_offsets_seconds"] = offsets
        manifest["seizures_by_recording"] = {
            name: [[event.onset_seconds, event.offset_seconds] for event in events]
            for name, events in annotations.seizures_by_file.items()
            if name in offsets
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)
        print(f"Added {len(horizons)} SOP label sets for {patient_id}.")

    return manifests
