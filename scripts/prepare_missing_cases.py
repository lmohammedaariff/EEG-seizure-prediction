"""Prepare CHB-MIT cases that do not yet have a cache manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seizure_prediction.config import ExperimentConfig
from seizure_prediction.edf import prepare_patient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/chbmit"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    args = parser.parse_args()

    existing_manifests = sorted(args.cache_dir.glob("chb*/manifest.json"))
    if not existing_manifests:
        raise SystemExit("No existing cache manifest found; run the regular prepare command first.")
    reference = json.loads(existing_manifests[0].read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(reference["config"])

    patients = sorted(
        path
        for path in args.data_dir.glob("chb*")
        if path.is_dir() and any(path.glob("*.edf"))
    )
    if not patients:
        raise SystemExit(f"No CHB-MIT EDF folders found in {args.data_dir}.")

    prepared = 0
    for index, patient in enumerate(patients, start=1):
        manifest_path = args.cache_dir / patient.name.lower() / "manifest.json"
        if manifest_path.is_file():
            continue
        if any(patient.glob("*.edf.part")):
            raise SystemExit(f"Incomplete EDF download found in {patient}; finish downloading first.")
        print(f"Preparing missing case {patient.name} ({index}/{len(patients)})", flush=True)
        prepare_patient(patient, args.cache_dir, config)
        prepared += 1
    print(f"Prepared {prepared} missing cases; total cached cases: {len(list(args.cache_dir.glob('chb*/manifest.json')))}")


if __name__ == "__main__":
    main()
