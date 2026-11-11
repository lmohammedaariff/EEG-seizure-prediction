from __future__ import annotations

import argparse
import concurrent.futures
import html
import os
import re
import shutil
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
DOWNLOAD_BASE_URL = "https://archive.physionet.org/physiobank/database/chbmit/"
CHUNK_BYTES = 1024 * 1024
RETRIES = 5
_progress_lock = threading.Lock()
_progress_bytes = 0
_last_report = 0.0
_progress_total = 0
_RESERVE_GIB = 15.0
_MAX_DOWNLOAD_GIB: float | None = 8.0


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None):
    request = Request(
        url,
        method=method,
        headers={"User-Agent": "seizure-prediction-research/0.1"} | (headers or {}),
    )
    return urlopen(request, timeout=90)


def _safe_relative_path(value: str) -> Path:
    posix = PurePosixPath(unquote(value))
    if posix.is_absolute() or not posix.parts or any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError(f"Unsafe path received from PhysioNet: {value!r}")
    return Path(*posix.parts)


def _fetch_file_list() -> list[str]:
    with _request(urljoin(BASE_URL, "RECORDS")) as response:
        text = response.read().decode("utf-8")
    edfs = [line.strip() for line in text.splitlines() if line.strip().lower().endswith(".edf")]
    if not edfs:
        raise RuntimeError("PhysioNet's CHB-MIT RECORDS index contained no EDF files.")
    patients = sorted({PurePosixPath(name).parts[0] for name in edfs})
    summaries = [f"{patient}/{patient}-summary.txt" for patient in patients]
    paths = sorted(set(edfs + summaries))
    for name in paths:
        _safe_relative_path(name)
    return paths


def _fetch_patient_sizes(patients: list[str]) -> dict[str, dict[str, int]]:
    """Read each official directory listing, avoiding hundreds of slow HEAD requests."""
    sizes: dict[str, dict[str, int]] = {}
    listing_pattern = re.compile(
        r'<a href="([^"]+)">[^<]*</a>\s+\d{2}-[A-Za-z]{3}-\d{4}'
        r'\s+\d{2}:\d{2}\s+(\d+)',
        re.IGNORECASE,
    )
    for patient in patients:
        url = urljoin(BASE_URL, f"{patient}/")
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                with _request(url) as response:
                    page = response.read().decode("utf-8", errors="replace")
                entries: dict[str, int] = {}
                for href, size_text in listing_pattern.findall(page):
                    name = html.unescape(href)
                    if not name.lower().endswith((".edf", ".txt")):
                        continue
                    relative_path = f"{patient}/{name}"
                    _safe_relative_path(relative_path)
                    entries[relative_path] = int(size_text)
                if not entries:
                    raise RuntimeError(f"No EEG or summary files listed for {patient}.")
                sizes[patient] = entries
                break
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < RETRIES:
                    time.sleep(min(2**attempt, 20))
        else:
            raise RuntimeError(
                f"Unable to read official file listing for {patient}: {last_error}"
            ) from last_error
    return sizes


def _select_complete_patients(
    patient_sizes: dict[str, dict[str, int]],
    target: Path,
    free_bytes: int,
    reserve_bytes: int,
    max_download_bytes: int | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Maximize the number of complete patients within the disk-space budget."""
    budget = free_bytes - reserve_bytes
    if max_download_bytes is not None:
        budget = min(budget, max_download_bytes)
    if budget < 0:
        raise RuntimeError(
            f"Less than the required {reserve_bytes / 1024**3:.1f} GiB is free at {target}."
        )

    quantum = 1024 * 1024
    states: dict[int, tuple[int, int, tuple[str, ...]]] = {0: (0, 0, ())}
    full_costs: dict[str, int] = {}
    for patient, files in patient_sizes.items():
        missing_bytes = 0
        for name, expected_size in files.items():
            destination = target / _safe_relative_path(name)
            partial = destination.with_name(destination.name + ".part")
            if destination.is_file():
                actual_size = destination.stat().st_size
                if actual_size != expected_size:
                    raise RuntimeError(
                        f"Existing file has unexpected size and will not be overwritten: "
                        f"{destination} ({actual_size} bytes; expected {expected_size})."
                    )
                continue
            if partial.is_file():
                actual_size = partial.stat().st_size
                if actual_size > expected_size:
                    raise RuntimeError(
                        f"Partial download is larger than the expected file: {partial}."
                    )
                missing_bytes += expected_size - actual_size
            else:
                missing_bytes += expected_size
        full_costs[patient] = missing_bytes

    capacity_units = budget // quantum
    for patient, missing_bytes in full_costs.items():
        weight = (missing_bytes + quantum - 1) // quantum
        if weight > capacity_units:
            continue
        updates = {}
        for used, (count, actual_bytes, selected) in states.items():
            new_used = used + weight
            if new_used > capacity_units:
                continue
            candidate = (count + 1, actual_bytes + missing_bytes, (*selected, patient))
            current = states.get(new_used)
            pending = updates.get(new_used)
            if pending is not None and (
                current is None
                or pending[0] > current[0]
                or (pending[0] == current[0] and pending[1] < current[1])
            ):
                current = pending
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and candidate[1] < current[1]
            ):
                updates[new_used] = candidate
        for used, candidate in updates.items():
            current = states.get(used)
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and candidate[1] < current[1]
            ):
                states[used] = candidate

    best = max(
        states.values(),
        key=lambda state: (state[0], state[1]),
    )
    return list(best[2]), full_costs


def _report_progress(path: str, transferred: int) -> None:
    global _progress_bytes, _last_report
    with _progress_lock:
        _progress_bytes += transferred
        now = time.monotonic()
        if now - _last_report < 4 and _progress_bytes < _progress_total:
            return
        _last_report = now
        percent = 100 * _progress_bytes / _progress_total if _progress_total else 100
        print(
            f"Progress: {_progress_bytes / 1024**3:.2f}/"
            f"{_progress_total / 1024**3:.2f} GiB "
            f"({percent:.1f}%) — {path}",
            flush=True,
        )


def _download_one(target: Path, relative_path: str, expected_size: int) -> tuple[str, bool]:
    destination = target / _safe_relative_path(relative_path)
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        existing_size = destination.stat().st_size
        if existing_size == expected_size:
            return relative_path, False
        raise RuntimeError(
            f"Existing file has unexpected size and will not be overwritten: {destination} "
            f"({existing_size} bytes; expected {expected_size})."
        )

    for attempt in range(RETRIES):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise RuntimeError(
                f"Partial download is larger than the expected file: {partial}."
            )
        request_headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with _request(
                urljoin(DOWNLOAD_BASE_URL, relative_path),
                headers=request_headers,
            ) as response:
                if offset and response.status == 206:
                    mode = "ab"
                else:
                    mode = "wb"
                    if offset:
                        offset = 0
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        stream.write(chunk)
                        _report_progress(relative_path, len(chunk))
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Downloaded {actual_size} bytes for {relative_path}; "
                    f"expected {expected_size}."
                )
            os.replace(partial, destination)
            return relative_path, True
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            if attempt + 1 == RETRIES:
                raise RuntimeError(f"Download failed for {relative_path}: {exc}") from exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Download failed for {relative_path}.")


def download_dataset(
    target: Path,
    reserve_gib: float = 15.0,
    workers: int = 8,
    max_download_gib: float | None = 8.0,
) -> None:
    if reserve_gib < 1:
        raise ValueError("Keep at least 1 GiB free on the destination drive.")
    if workers < 1 or workers > 8:
        raise ValueError("Use between 1 and 8 download workers.")
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    print("Reading the official PhysioNet CHB-MIT file index...", flush=True)
    all_paths = _fetch_file_list()
    patients = sorted({PurePosixPath(name).parts[0] for name in all_paths})
    print(f"Checking official directory listings for {len(patients)} patients...", flush=True)
    patient_sizes = _fetch_patient_sizes(patients)
    free_bytes = shutil.disk_usage(target).free
    reserve_bytes = int(reserve_gib * 1024**3)
    max_download_bytes = (
        int(max_download_gib * 1024**3) if max_download_gib is not None else None
    )
    selected_patients, full_costs = _select_complete_patients(
        patient_sizes,
        target,
        free_bytes,
        reserve_bytes,
        max_download_bytes,
    )
    if len(selected_patients) < 4:
        raise RuntimeError(
            f"With a {reserve_gib:.1f} GiB reserve, only {len(selected_patients)} "
            f"complete subjects fit at {target}; at least four are needed for "
            "patient-wise training, validation, and testing. Free more disk space."
        )
    excluded = sorted(set(patients) - set(selected_patients))
    sizes = {
        name: size
        for patient in selected_patients
        for name, size in patient_sizes[patient].items()
    }
    paths = sorted(sizes)
    remaining_bytes = sum(full_costs[patient] for patient in selected_patients)
    total_size = sum(sizes.values())
    if excluded:
        print(
            f"Space limit: downloading {len(selected_patients)}/{len(patients)} "
            "complete subjects to preserve the requested free-space reserve.",
            flush=True,
        )
    print(f"Included complete subjects: {', '.join(selected_patients)}", flush=True)
    if excluded:
        print(f"Not downloaded: {', '.join(excluded)}", flush=True)
    print(
        f"Preflight passed: {len(paths)} files from {len(selected_patients)} "
        f"subjects, {total_size / 1024**3:.2f} GiB total; "
        f"{remaining_bytes / 1024**3:.2f} GiB to download, "
        f"{free_bytes / 1024**3:.2f} GiB free at {target}.",
        flush=True,
    )
    global _progress_bytes, _last_report, _progress_total
    _progress_bytes = 0
    _progress_total = remaining_bytes
    _last_report = 0.0
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, target, name, sizes[name]): name
            for name in paths
        }
        for future in concurrent.futures.as_completed(futures):
            name, downloaded = future.result()
            completed += 1
            action = "Downloaded" if downloaded else "Already present"
            print(
                f"[{completed}/{len(paths)}] {action}: {name}",
                flush=True,
            )
    print(f"CHB-MIT is ready at {target}.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CHB-MIT directly from its official PhysioNet archive."
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help=r"Destination folder, e.g. D:\datasets\chbmit.",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=15.0,
        help="Minimum free space to leave on the target drive (default: 15 GiB).",
    )
    parser.add_argument(
        "--max-download-gib",
        type=float,
        default=8.0,
        help="Cap new downloads to leave room for preprocessing (default: 8 GiB).",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download_dataset(
        args.target,
        reserve_gib=args.reserve_gib,
        workers=args.workers,
        max_download_gib=args.max_download_gib,
    )


if __name__ == "__main__":
    main()
