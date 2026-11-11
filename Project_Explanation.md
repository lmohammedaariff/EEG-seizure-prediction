# Project guide

## Explain the project in one minute

This project explores whether multichannel scalp EEG can identify seizure risk before a seizure. It trains a shared model to estimate preictal risk at 5, 10, 15, and 30-minute seizure-onset prediction horizons, using a fixed 5-minute seizure prediction horizon. It compares the proposed multi-domain model with CNN, LSTM, and CNN-LSTM baselines, evaluates held-out people, measures event sensitivity and false alarms per hour, and displays model explanations in a React + FastAPI dashboard.

## What is distinctive in this study

The project combines a shared multi-horizon EEG model with patient-held-out evaluation, event-level alarm metrics, matched architecture comparisons, and explanation stability/perturbation audits. It does not claim that CNN, LSTM, attention, or explainability alone is new. The study tests whether these combined choices improve forecasting across people while showing the sensitivity/false-alarm trade-off.

## How to start the dashboard

Double-click `Run_Project.bat` from this folder, keep the terminal open, and open `http://127.0.0.1:8000`. To stop the server, press Ctrl+C in that terminal. The launcher uses the E: project environment when available.

## Where to find things

| Folder or file | What it contains |
|---|---|
| `data/chbmit/` | Downloaded CHB-MIT EEG recordings and seizure summaries. Keep these files. |
| `data/siena/` | Downloaded Siena EEG recordings and checksum/annotation metadata. Keep these files. |
| `artifacts/results-multihorizon-16case/` | Trained CHB-MIT checkpoints, LOSO metrics, explanation audits, and the main report. |
| `artifacts/results-siena-fullcohort/` | External Siena evaluation outputs and per-participant results. |
| `artifacts/results-multihorizon/` and `artifacts/results/` | Earlier experiment outputs retained for reference; use the `*-16case` folder for the main corrected result. |
| `src/seizure_prediction/` | EEG preprocessing, labels, model architectures, training, evaluation, and prediction code. |
| `frontend/` | React dashboard source and its built files. |
| `api_server.py` | FastAPI backend used by the dashboard. |
| `run_project.py` and `Run_Project.bat` | Start the dashboard. |
| `scripts/` | Dataset download, training/evaluation, reporting, and packaging commands. |
| `docs/` | Architecture diagram, research-gap review, sensitivity review, and proposal checklist. |
| `deliverables/` | `Old_Project_Backup.zip` is an older full archive; `Project_Guide_and_Siena_Results.zip` is the small external-results supplement. |
| `.venv-gpu/` | Local Python environment. Do not delete if you want to run the project without reinstalling packages. |
| `tmp_uploads/` | Temporary files used while the dashboard analyzes uploads. The app recreates this folder if needed. |

## Current evidence and limits

CHB-MIT experiments cover 16 case folders representing 15 people, not all cases in the full database. The full locally downloaded Siena cohort has 14 participants, 41 EDF recordings, and 47 annotated seizures. The external test uses one frozen CHB-MIT fold per model. On Siena, CHB-calibrated thresholds did not preserve the desired sensitivity/false-alarm balance. These are retrospective research results, not clinical validation or a medical warning system.

For the detailed methods and commands, see `README.md`. For the cross-dataset results, see `artifacts/results-siena-fullcohort/siena_external_report.md`.
