from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import (
    MULTIHORIZON_MODEL_VARIANTS,
    SINGLE_HORIZON_VARIANTS,
    ExperimentConfig,
)
from .edf import prepare_dataset, relabel_cache_horizons
from .evaluation import evaluate_loso
from .multihorizon_eval import evaluate_multihorizon_loso
from .predict import predict_edf
from .datasets import MultiHorizonEEGDataset
from .training import train_final_multihorizon_model


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Horizons must be comma-separated integers.") from exc
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("Provide one or more positive horizons.")
    return horizons


def _read_cache_config(cache_dir: Path) -> ExperimentConfig:
    manifests = sorted(cache_dir.glob("chb*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No prepared patient manifests found under {cache_dir}. Run 'prepare' first."
        )
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(payload["config"])


def _add_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seizure-prediction",
        description="CHB-MIT patient-independent seizure prediction research pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Preprocess EDFs into cached windows.")
    prepare.add_argument("--data-dir", type=Path, default=Path("data/chbmit"))
    prepare.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    prepare.add_argument("--horizons", type=_parse_horizons, default=(5, 10, 15, 30))
    prepare.add_argument("--sph-minutes", type=int, default=5)
    prepare.add_argument("--interictal-buffer-minutes", type=int, default=60)
    prepare.add_argument("--window-seconds", type=float, default=10.0)
    prepare.add_argument("--overlap", type=float, default=0.5)
    prepare.add_argument("--max-recordings-per-patient", type=int)

    relabel = subparsers.add_parser(
        "relabel",
        help="Add SOP-specific labels to an existing prepared cache without reprocessing EEG.",
    )
    relabel.add_argument("--data-dir", type=Path, default=Path("data/chbmit"))
    relabel.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    relabel.add_argument("--horizons", type=_parse_horizons, default=(5, 10, 15, 30))

    evaluate = subparsers.add_parser(
        "evaluate", help="Run single-horizon LOSO ablations and baselines."
    )
    evaluate.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    evaluate.add_argument("--horizons", type=_parse_horizons)
    evaluate.add_argument("--models", nargs="+", choices=SINGLE_HORIZON_VARIANTS)
    evaluate.add_argument("--include-baselines", action="store_true")
    evaluate.add_argument("--max-folds", type=int)
    evaluate.add_argument("--epochs", type=int, default=10)
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument(
        "--samples-per-epoch",
        type=int,
        help="Optional cap on weighted training samples per epoch.",
    )

    multihorizon = subparsers.add_parser(
        "evaluate-multihorizon",
        help="Train shared horizon heads and evaluate validation-calibrated event alarms.",
    )
    multihorizon.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    multihorizon.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    multihorizon.add_argument(
        "--models", nargs="+", choices=MULTIHORIZON_MODEL_VARIANTS
    )
    multihorizon.add_argument("--max-folds", type=int)
    multihorizon.add_argument("--epochs", type=int, default=6)
    multihorizon.add_argument("--batch-size", type=int, default=64)
    multihorizon.add_argument("--samples-per-epoch", type=int, default=6000)
    multihorizon.add_argument("--threshold-candidates", type=int, default=81)
    multihorizon.add_argument(
        "--resume",
        action="store_true",
        help="Continue compatible LOSO folds from existing checkpoints and results.",
    )

    final_fit = subparsers.add_parser(
        "fit-multihorizon-all",
        help="Fit a final checkpoint on every labeled window in the cohort.",
    )
    final_fit.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    final_fit.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/results-multihorizon-16case/final_model/model.pt"),
    )
    final_fit.add_argument(
        "--model",
        choices=MULTIHORIZON_MODEL_VARIANTS,
        default="multi_horizon",
    )
    final_fit.add_argument("--epochs", type=int, default=3)
    final_fit.add_argument("--batch-size", type=int, default=64)
    final_fit.add_argument(
        "--samples-per-epoch",
        type=int,
        default=6000,
        help="Patient/state-balanced windows per epoch to bound memory use.",
    )

    predict = subparsers.add_parser("predict", help="Predict EEG windows in an EDF.")
    predict.add_argument("--edf", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("prediction.csv"))
    predict.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = _add_parser().parse_args()
    if args.command == "prepare":
        config = ExperimentConfig(
            preictal_minutes=max(args.horizons),
            sph_minutes=args.sph_minutes,
            interictal_buffer_minutes=args.interictal_buffer_minutes,
            horizons_minutes=args.horizons,
            window_seconds=args.window_seconds,
            overlap=args.overlap,
        )
        manifests = prepare_dataset(
            args.data_dir,
            args.cache_dir,
            config,
            max_recordings_per_patient=args.max_recordings_per_patient,
        )
        print(f"Prepared {len(manifests)} patients in {args.cache_dir}.")
    elif args.command == "relabel":
        manifests = relabel_cache_horizons(
            args.data_dir,
            args.cache_dir,
            args.horizons,
        )
        print(f"Relabeled {len(manifests)} patients for SOPs {args.horizons}.")
    elif args.command == "evaluate":
        config = _read_cache_config(args.cache_dir)
        config = replace(
            config,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results = evaluate_loso(
            args.cache_dir,
            args.output_dir,
            config,
            model_variants=tuple(args.models) if args.models else None,
            horizons=args.horizons,
            max_folds=args.max_folds,
            include_baselines=args.include_baselines,
            samples_per_epoch=args.samples_per_epoch,
        )
        print(f"Completed {len(results)} held-out patient evaluations.")
        print(f"Summary: {args.output_dir / 'loso_summary.csv'}")
    elif args.command == "evaluate-multihorizon":
        config = _read_cache_config(args.cache_dir)
        config = replace(
            config,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results = evaluate_multihorizon_loso(
            args.cache_dir,
            args.output_dir,
            config,
            model_variants=(
                tuple(args.models)
                if args.models
                else MULTIHORIZON_MODEL_VARIANTS
            ),
            max_folds=args.max_folds,
            samples_per_epoch=args.samples_per_epoch,
            threshold_candidates=args.threshold_candidates,
            resume=args.resume,
        )
        print(f"Completed {len(results)} patient/horizon/alarm-budget evaluations.")
        print(f"Summary: {args.output_dir / 'multihorizon_loso_summary.csv'}")
    elif args.command == "fit-multihorizon-all":
        config = replace(
            _read_cache_config(args.cache_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_variant=args.model,
        )
        manifests = sorted(args.cache_dir.glob("chb*/manifest.json"))
        train_datasets = [
            MultiHorizonEEGDataset(
                manifest.parent,
                config.horizons_minutes,
                subject_index=index,
            )
            for index, manifest in enumerate(manifests)
        ]
        if len(train_datasets) < 2:
            raise ValueError("Final multi-horizon fit requires at least two patients.")
        for dataset in train_datasets:
            print(
                f"{dataset.patient_id}: {len(dataset.recording_names)} labeled EDFs, "
                f"{len(dataset):,} windows"
            )
        result = train_final_multihorizon_model(
            train_datasets,
            config,
            args.model,
            args.checkpoint,
            samples_per_epoch=args.samples_per_epoch,
        )
        print(
            f"Completed final fit on {result['training_recordings']} labeled EDFs "
            f"and {result['training_windows']:,} available windows; sampled "
            f"{result['training_samples_per_epoch']:,} per epoch "
            f"({result['device']})."
        )
        print(f"Checkpoint: {result['checkpoint']}")
    elif args.command == "predict":
        frame, _, _, _, _ = predict_edf(
            args.edf,
            args.checkpoint,
            batch_size=args.batch_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"Saved {len(frame)} window predictions to {args.output}.")


if __name__ == "__main__":
    main()
