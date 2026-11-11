"""Build a current project ZIP, including its datasets and results, on E:."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "deliverables" / "Current_Project_Complete.zip"
ARCHIVE_ROOT = "SeizurePredictionResearch"
INCLUDE_PATHS = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "api_server.py",
    "run_project.py",
    "Run_Project.bat",
    "Project_Explanation.md",
    "docs",
    "scripts",
    "src",
    "tests",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/index.html",
    "frontend/src",
    "frontend/dist",
    "data/chbmit",
    "data/siena/SHA256SUMS.txt",
    "data/siena/LICENSE.txt",
    "data/siena/siena_subset_manifest.json",
    "data/siena/subject_info.csv",
    "artifacts/results-multihorizon-16case",
    "artifacts/results-siena-fullcohort",
)
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "deliverables",
    "torchinductor_Admin",
}
EXCLUDED_NAMES = {".env", ".env.local", "secrets.toml"}


def included_files() -> list[Path]:
    files: set[Path] = set()
    for relative in INCLUDE_PATHS:
        source = ROOT / relative
        if not source.exists():
            continue
        candidates = [source] if source.is_file() else source.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            parts = path.relative_to(ROOT).parts
            if any(part in EXCLUDED_PARTS for part in parts):
                continue
            if path.name.lower() in EXCLUDED_NAMES or path.suffix.lower() in {".pyc", ".pyo", ".log"}:
                continue
            files.add(path)
    return sorted(files)


def create_archive(destination: Path) -> tuple[int, int, str]:
    destination = destination.resolve()
    if destination.drive.lower() != "e:":
        raise ValueError("Save the project bundle on the E: drive.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    paths = included_files()
    raw_bytes = sum(path.stat().st_size for path in paths)
    digest = hashlib.sha256()
    with ZipFile(partial, "w", compression=ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for index, path in enumerate(paths, start=1):
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{ARCHIVE_ROOT}/{relative}")
            digest.update(relative.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            if index % 200 == 0 or index == len(paths):
                print(f"Archived {index}/{len(paths)} files", flush=True)
    with ZipFile(partial) as archive:
        corrupt = archive.testzip()
        if corrupt:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"ZIP verification failed at {corrupt}")
    os.replace(partial, destination)
    return len(paths), raw_bytes, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count, raw_bytes, manifest_digest = create_archive(args.output)
    print(
        f"Created and verified {args.output}: {count} files, "
        f"{raw_bytes / (1024 ** 3):.2f} GiB before compression, "
        f"{args.output.stat().st_size:,} bytes compressed.",
        flush=True,
    )
    print(f"Bundle manifest digest: {manifest_digest}", flush=True)


if __name__ == "__main__":
    main()
