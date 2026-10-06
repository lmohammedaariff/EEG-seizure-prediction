from __future__ import annotations
import argparse
import json
import copy
from pathlib import Path
import pandas as pd
from .config import load_config
from .train import run_loso, STAGES, BASELINES
from .data import load_dataset
from .siena import load_siena_dataset, evaluate_siena


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the proposal-aligned EEG framework with LOSO.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--horizons", nargs="+", type=int, help="Subset of 5 10 15 30; default uses configured values")
    parser.add_argument("--stages", nargs="+", choices=STAGES + BASELINES,
                        default=["cnn", "lstm", "cnn_lstm", *STAGES])
    parser.add_argument("--quick", action="store_true", help="One horizon and complete model only")
    parser.add_argument("--external-siena", action="store_true",
                        help="After CHB-MIT LOSO, evaluate its fold checkpoints on compatible Siena EDFs")
    parser.add_argument("--exclusion-sensitivity", action="store_true",
                        help="Sweep configured interictal and postictal exclusion buffers separately")
    args = parser.parse_args()
    cfg = load_config(args.config)
    horizons = args.horizons or cfg["data"]["preictal_horizons_minutes"]
    if any(h not in (5, 10, 15, 30) for h in horizons):
        parser.error("horizons must be selected from 5, 10, 15, 30")
    stages = args.stages
    if args.quick:
        horizons, stages = [horizons[0]], ["complete"]
    all_metrics = []
    base_output = Path(cfg["data"]["output"])
    if args.exclusion_sensitivity:
        cases = [(axis, int(value)) for axis in ("interictal", "postictal")
                 for value in cfg["data"].get("buffer_sensitivity_seconds", [0, 60, 300])]
    else:
        cases = [("configured", None)]
    for horizon in horizons:
        for axis, value in cases:
            run_cfg = copy.deepcopy(cfg)
            if value is not None:
                key = f"{axis}_exclusion_seconds"
                run_cfg["data"][key] = value
                case_name = f"{axis}_{value}s"
                run_cfg["data"]["output"] = str(base_output / "exclusion_sensitivity" / case_name)
                print(f"Sensitivity run: {axis} exclusion = {value} seconds; horizon = {horizon} minutes")
            else:
                case_name = "configured"
                print(f"Loading CHB-MIT for the {horizon}-minute horizon...")
            run_output = Path(run_cfg["data"]["output"])
            metadata_path = run_output / f"window_labels_{horizon}m.csv"
            dataset = load_dataset(run_cfg, horizon, metadata_path=metadata_path)
            siena_dataset = None
            if args.external_siena:
                print("Loading optional Siena data; every EDF must support the standardized CHB-MIT bipolar montage...")
                siena_metadata = run_output / f"siena_window_labels_{horizon}m.csv"
                siena_dataset = load_siena_dataset(run_cfg, horizon, metadata_path=siena_metadata)
            for stage in stages:
                frame, summary = run_loso(run_cfg, horizon, stage, dataset=dataset)
                if not frame.empty:
                    frame["exclusion_axis"] = axis
                    frame["exclusion_seconds"] = value
                    all_metrics.append(frame)
                    print(json.dumps({"horizon_minutes": horizon, "stage": stage,
                                      "exclusion_axis": axis, "exclusion_seconds": value, **summary}, indent=2))
                if args.external_siena:
                    _, external_summary = evaluate_siena(
                        run_cfg, horizon, stage, run_output / f"horizon_{horizon}m" / stage,
                        dataset=siena_dataset)
                    print(json.dumps({"external_dataset": "Siena", "horizon_minutes": horizon,
                                      "stage": stage, "exclusion_axis": axis,
                                      "exclusion_seconds": value, **external_summary}, indent=2))
            del dataset
    if all_metrics:
        base_output.mkdir(parents=True, exist_ok=True)
        name = "exclusion_sensitivity_patient_metrics.csv" if args.exclusion_sensitivity else "all_loso_patient_metrics.csv"
        pd.concat(all_metrics, ignore_index=True).to_csv(base_output / name, index=False)
        print(f"Patient-wise results saved under {base_output.resolve()}")


if __name__ == "__main__":
    main()
