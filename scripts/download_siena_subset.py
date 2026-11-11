from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, unquote, urljoin
from urllib.request import Request, urlopen


BASE = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/"
CONTENT_BASE = "https://physionet.org/content/siena-scalp-eeg/1.0.0/"
DEFAULT_SUBJECTS = (
    "PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09",
    "PN10", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17",
)
RESERVE_BYTES = 512 * 1024 * 1024


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "SeizurePredictionResearch/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def list_subject_edfs(
    subject: str, size_cache_path: Path | None = None
) -> list[tuple[str, str, int]]:
    page = fetch(urljoin(CONTENT_BASE, f"{subject}/")).decode("utf-8", "replace")
    rows = re.findall(
        r'href="([^"?]+\.edf)"[^>]*>([^<]+)</a>.*?<td>\s*([\d.]+)\s*(MB|GB)\s*</td>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    items = []
    for href, visible_name, value, unit in rows:
        # The HTML directory listing rounds sizes (occasionally by >1%), so it
        # is suitable for discovery only. Fetch exact object lengths below.
        items.append((Path(unquote(href)).name, 0))
    if not items:
        raise RuntimeError(f"Could not read EDF file listing for {subject} from PhysioNet.")

    def exact_size(item: tuple[str, int]) -> tuple[str, int]:
        name, _ = item
        url = urljoin(BASE, f"{subject}/{quote(name)}")
        request = Request(url, method="HEAD", headers={"User-Agent": "SeizurePredictionResearch/1.0"})
        with urlopen(request, timeout=60) as response:
            header = response.headers.get("Content-Length")
        if header is None:
            raise RuntimeError(f"PhysioNet did not provide exact size for {subject}/{name}.")
        return name, int(header)

    cached_sizes: dict[str, dict[str, int]] = {}
    if size_cache_path and size_cache_path.is_file():
        cached_sizes = json.loads(size_cache_path.read_text(encoding="utf-8"))
    cached_subject = cached_sizes.get(subject, {})
    missing = [item for item in items if item[0] not in cached_subject]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as pool:
            resolved = list(pool.map(exact_size, missing))
        cached_subject.update(dict(resolved))
        cached_sizes[subject] = cached_subject
        if size_cache_path:
            size_cache_path.parent.mkdir(parents=True, exist_ok=True)
            size_cache_path.write_text(json.dumps(cached_sizes, indent=2), encoding="utf-8")
    exact = [(name, int(cached_subject[name])) for name, _ in items]
    return [(name, size) for name, size in exact]


def download_file(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise RuntimeError(f"Partial download exceeds the exact object size: {partial}.")
    if expected_size > 0 and offset == expected_size:
        partial.replace(destination)
        return
    headers = {"User-Agent": "SeizurePredictionResearch/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=90) as response:
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                print(
                    f"  {destination.name}: {partial.stat().st_size / 1024**2:.0f} MiB",
                    flush=True,
                )
    if abs(partial.stat().st_size - expected_size) > max(1024 * 1024, expected_size * 0.012):
        raise RuntimeError(
            f"Downloaded {partial.stat().st_size} bytes for {destination.name}; "
            f"the PhysioNet listing reports {expected_size}. Partial file retained for resume."
        )
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the selected Siena external-validation cohort from PhysioNet."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/siena"))
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--reserve-gb", type=float, default=0.5)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.reserve_gb < 0:
        raise ValueError("--reserve-gb cannot be negative.")

    files: list[tuple[str, str, int]] = []
    exact_size_cache = args.data_dir / ".physionet_exact_file_sizes.json"
    for subject in args.subjects:
        files.extend(
            (subject, *item)
            for item in list_subject_edfs(subject.upper(), exact_size_cache)
        )
        annotation = f"Seizures-list-{subject.upper()}.txt"
        files.append((subject.upper(), annotation, 0))
    for name in ("subject_info.csv", "LICENSE.txt", "SHA256SUMS.txt"):
        files.append(("", name, 0))

    required = 0
    for subject, name, expected_size in files:
        if not expected_size:
            continue
        destination = args.data_dir / subject / name
        partial = destination.with_suffix(destination.suffix + ".part")
        if destination.is_file():
            if destination.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Existing file size mismatch for {destination}: "
                    f"{destination.stat().st_size} != {expected_size}."
                )
            continue
        partial_size = partial.stat().st_size if partial.is_file() else 0
        if partial_size > expected_size:
            raise RuntimeError(f"Partial download exceeds the exact object size: {partial}.")
        required += expected_size - partial_size
    drive = args.data_dir.anchor or str(args.data_dir)
    usage = shutil.disk_usage(drive)
    reserve = max(RESERVE_BYTES, int(args.reserve_gb * 1024**3))
    print(f"Siena subjects: {', '.join(s.upper() for s in args.subjects)}")
    print(f"Bytes still needed: {required:,}; free bytes: {usage.free:,}; reserved: {reserve:,}")
    if required + reserve > usage.free:
        raise RuntimeError("Insufficient free space; reduce subjects or free space on the target drive.")

    manifest: dict[str, object] = {
        "source": "https://physionet.org/content/siena-scalp-eeg/1.0.0/",
        "version": "1.0.0",
        "subjects": [subject.upper() for subject in args.subjects],
        "downloaded": [],
    }
    for subject, name, expected_size in files:
        relative = Path(subject) / name if subject else Path(name)
        destination = args.data_dir / relative
        url = urljoin(BASE, f"{subject}/{name}" if subject else name)
        size_ok = expected_size == 0 or abs(destination.stat().st_size - expected_size) <= max(
            1024 * 1024, expected_size * 0.012
        ) if destination.is_file() else False
        if destination.is_file() and size_ok:
            print(f"Already present: {relative}")
        else:
            print(f"Downloading {relative}")
            download_file(url, destination, expected_size)
        manifest["downloaded"].append(
            {"path": relative.as_posix(), "bytes": destination.stat().st_size}
        )
    checksum_file = args.data_dir / "SHA256SUMS.txt"
    checksums: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$", line)
        if match:
            checksums[Path(match.group(2)).as_posix().lower()] = match.group(1).lower()
    verified = []
    for item in manifest["downloaded"]:
        relative = Path(item["path"])
        if relative.suffix.lower() != ".edf":
            continue
        checksum_key = relative.as_posix().lower()
        expected_hash = checksums.get(checksum_key)
        if expected_hash is None:
            raise RuntimeError(f"No official SHA256 checksum was found for {relative}.")
        digest = hashlib.sha256()
        with (args.data_dir / relative).open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise RuntimeError(f"Official SHA256 checksum mismatch for {relative}.")
        verified.append(relative.as_posix())
    manifest["sha256_verified_edfs"] = verified
    (args.data_dir / "siena_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Ready: {args.data_dir}")


if __name__ == "__main__":
    main()
