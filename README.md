# Patient-independent multi-horizon seizure forecasting with multi-domain EEG learning

An end-to-end research project for CHB-MIT scalp EEG: reproducible EDF preprocessing and SOP/SPH labeling, one shared EEG representation with horizon-specific prediction heads, temporal/spectral/time-frequency feature learning, attention fusion, adversarial subject-invariant learning, strict patient-held-out validation, event-level alarm evaluation, matched CNN/LSTM baselines, explanation audits, and a React + FastAPI interface.

> **Research use only.** This is not a medical device and must not be used to diagnose, treat, or make clinical decisions. A preictal score is an experimental model output, not a guaranteed warning.

## Project documents and results

- [Project abstract (PDF)](deliverables/Abstract_Mohammed_Aariff_L.pdf)
- [Project architecture (PDF)](deliverables/Project_Architecture_Seizure_Prediction.pdf) | [Word document](deliverables/Project_Architecture_Seizure_Prediction.docx)
- [CHB-MIT LOSO results](artifacts/results-multihorizon-16case/final_results_report.md)
- [Statistical uncertainty and paired comparisons](artifacts/results-multihorizon-16case/statistical_uncertainty_report.md)
- [Siena external evaluation](artifacts/results-siena-fullcohort/siena_external_report.md)
- [Evaluation integrity audit](docs/evaluation_integrity_audit.md)
- [Dashboard running status](RUN_STATUS.txt)
- [Download the repository source and outputs as a ZIP](https://github.com/lmohammedaariff/EEG-seizure-prediction/archive/refs/heads/main.zip)

## What is implemented

- CHB-MIT summary parsing and EDF preprocessing (60 Hz notch, 0.5–45 Hz band-pass, 128 Hz resampling, fixed bipolar channel order).
- Three-state Interictal / Preictal / Ictal labels and a separate preictal-versus-interictal prediction task. Ictal windows are not treated as preictal negatives.
- Temporal CNN, PSD band-power MLP, STFT time-frequency CNN, learned gated attention fusion, and gradient-reversal subject adversary.
- Temporal-only, temporal+spectral, full multi-domain, full+subject-invariant ablations; CNN, LSTM, and CNN-LSTM baselines.
- Subject-level train/validation/test separation and LOSO folds, with per-subject results, mean/standard deviation, sensitivity, specificity, F1, and PR-AUC. CHB-MIT cases chb01 and chb21 are recognized as the same person and kept together in every split.
- Imbalance handling with patient/state-balanced sampling, per-horizon class-weighted loss, and a per-patient/per-horizon label-count audit.
- A shared multi-horizon model with horizon-specific heads for 5-, 10-, 15-, and 30-minute SOPs; matched multi-head CNN/LSTM/CNN-LSTM comparisons.
- A matched multi-horizon ablation sequence: temporal-only, temporal+spectral, full multi-domain attention, then subject-invariant adversarial learning.
- Event-level sensitivity, validation-only threshold selection at explicit false-alarms-per-hour budgets, alarm merging, interictal exposure accounting, and seizure lead-time reporting.
- Integrated Gradients, channel attribution, and fusion-domain attention inspection.
- React frontend with a FastAPI backend for EDF upload and inference, LOSO summaries, and the generated research report.
- Live dashboard charts for patient-held-out event sensitivity and observed false alarms by horizon; EDF analysis plots the complete recording's risk curve plus real EEG traces, relative power spectrum, and short-time Fourier representation from the peak-score window. You can select an EDF already downloaded under `data/chbmit/` or upload a different EDF.
- Architecture diagram: [`docs/architecture.svg`](docs/architecture.svg).

## Quick start (Windows PowerShell)

The supported user interface is React + FastAPI. To run this prepared E-drive
project now, double-click `Run_Project.bat`; its terminal window must stay open
while you use the dashboard. The commands below are for a fresh environment.

```powershell
$env:TMP = "$PWD\tmp_uploads"
$env:TEMP = $env:TMP
$env:PIP_CACHE_DIR = "$PWD\artifacts\pip-cache"
New-Item -ItemType Directory -Force $env:TMP, $env:PIP_CACHE_DIR | Out-Null
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python run_project.py
```

Open `http://127.0.0.1:8000`. The full backup ZIP includes the built React
frontend, so Node/npm are not required to run the app. The prepared E-drive
project uses `.venv-gpu`; a fresh installation can use `.venv`. The launcher starts the local-only FastAPI service, which
serves the React UI and the `/api/*` endpoints. Keep its terminal open while
using the dashboard; press Ctrl+C to stop the service.

## Dataset setup

This E-drive project currently includes 16 complete CHB-MIT case folders:
`chb01`, `chb02`, `chb03`, `chb05`, `chb08`, `chb11`, `chb13`, `chb14`,
`chb16`, `chb17`, `chb19`, `chb20`, `chb21`, `chb22`, `chb23`, and `chb24`.
Cases `chb01` and `chb21` are from the same person and are grouped together in
all participant splits. The official CHB-MIT dataset is substantially larger;
this project's results apply only to the included subset.

Download the **CHB-MIT Scalp EEG Database** from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) and extract it into `data/chbmit/`. Keep each patient's `.edf` files and matching `chbXX-summary.txt` together in the patient directory:

```text
data/chbmit/
  chb01/
    chb01-summary.txt
    chb01_01.edf
    ...
  chb02/
    chb02-summary.txt
```

To download the official EDF recordings and seizure-summary annotations directly from PhysioNet, use the resumable downloader:

```powershell
python scripts\download_chbmit.py --target E:\SeizurePredictionResearch\data\chbmit --reserve-gib 15 --max-download-gib 8 --workers 8
```

The downloader checks the live official file sizes and available disk space before transferring anything, verifies every downloaded file, and resumes interrupted `.part` files. It selects complete patient folders within your chosen download limit; check available storage for both raw EDFs and the preprocessed cache. The current E-drive project and rebuilt ZIP include the 16 case folders listed above. Results apply only to this subset.

### Siena external evaluation

The full locally downloaded Siena cohort is stored in `data/siena/`: 14 participants, 41 EDF recordings, and 47 annotated seizure events. The downloader can resume/verify files from PhysioNet:

```powershell
python scripts\download_siena_subset.py --data-dir data\siena
```

Run the frozen CHB-MIT held-out-fold models on Siena using each checkpoint's own threshold calibrated on that fold's separate CHB-MIT validation patient:

```powershell
$env:PYTHONPATH = "src"
python scripts\evaluate_siena.py --data-dir data\siena --results-dir artifacts\results-multihorizon-16case --fold chb08 --variants multi_horizon multi_horizon_cnn multi_horizon_lstm multi_horizon_cnn_lstm
```

The completed evaluation is in `artifacts/results-siena-fullcohort/`, including per-subject metrics, pooled summaries, annotation audit, manifest, and report. Frozen checkpoints from one CHB-MIT `chb08` LOSO fold were evaluated at 5-, 10-, 15-, and 30-minute SOPs using thresholds selected only on that fold's CHB validation participant. At 30-minute SOP, the proposed model detected 0/47 events at its strict 0.1/h threshold; at its 0.5/h threshold it detected 45/47 but yielded 9.13 false alarms per eligible interictal hour. The matched LSTM and CNN-LSTM detected 47/47 at the 0.5/h threshold, with 0.70 and 0.66 observed false alarms/hour respectively. These results show the CHB threshold does not transfer reliably. This is external dataset evaluation, not clinical validation or average performance across all CHB folds. See `docs/research_gap.md` and `docs/sensitivity_review.md`.

The two-consecutive-window alert-persistence analysis is also available for the proposed multi-horizon model and LSTM baseline. It uses the same separate validation-patient protocol and writes results under a distinct suffix, preserving the original metrics:

```powershell
$env:PYTHONPATH = "src"
python scripts\rebuild_multihorizon_results.py --cache-dir artifacts\cache --output-dir artifacts\results-multihorizon-16case --samples-per-epoch 6000 --alarm-persistence-windows 2 --output-suffix persistence2 --variants multi_horizon multi_horizon_lstm
python scripts\summarize_patient_level_results.py --results-dir artifacts\results-multihorizon-16case --input-suffix persistence2 --output-suffix persistence2
```

The persistence gate is an alert-policy experiment, not an architecture change. Its effect on sensitivity and FAR is reported separately from the one-window default.

URL-escaped filenames from PhysioNet are decoded before writing them locally
(for example, `chb02_16%2B.edf` becomes `chb02_16+.edf`). Preprocessing checks
that summary-referenced recordings exist. Recording timelines use EDF-header
start timestamps when available, rather than assuming alphabetical filename
order; missing annotated recordings or cache/EDF mismatches stop relabeling.

## Prepare and train

Preprocessing writes one memory-mapped-friendly `.npy` window/label pair per EDF to `artifacts/cache/`. This cache can be large; preprocessing one patient at a time avoids keeping the complete EEG collection in RAM.

```powershell
python -m seizure_prediction.cli prepare --data-dir data\chbmit --cache-dir artifacts\cache
python -m seizure_prediction.cli evaluate --cache-dir artifacts\cache --output-dir artifacts\results --epochs 10 --include-baselines
```

For a practical primary-model run, train the subject-invariant model at a 30-minute preictal horizon across every prepared LOSO fold:

```powershell
python -m seizure_prediction.cli evaluate --cache-dir artifacts\cache --output-dir artifacts\results-single-horizon --horizons 30 --models subject_invariant --epochs 10 --samples-per-epoch 12000
```

### Shared multi-horizon research experiment

If starting from the original 30-minute-only prepared cache, the following
relabels windows for 5-, 10-, 15-, and 30-minute SOPs directly from the CHB-MIT
summaries without rereading/resampling EEG signals. It refreshes recording
offsets from EDF headers and checks the raw/cache file sets, then trains
the shared multi-horizon ablation sequence and matched multi-head CNN, LSTM, and
CNN-LSTM baselines:

```powershell
python -m seizure_prediction.cli relabel --data-dir data\chbmit --cache-dir artifacts\cache --horizons 5,10,15,30
python -m seizure_prediction.cli evaluate-multihorizon --cache-dir artifacts\cache --output-dir artifacts\results-multihorizon-16case --epochs 6 --samples-per-epoch 6000
python scripts\write_multihorizon_report.py
```

The outer LOSO test patient is excluded from training and model selection. A
different patient is used for validation; only that validation patient's scores
select thresholds that maximize event sensitivity under 0.1, 0.5, and 1.0 false
alarms per interictal recording hour. Thresholds are then frozen for the test
patient. The event forecast interval is
`[seizure onset - SOP - SPH, seizure onset - SPH)`. Consecutive high-score
windows are merged using the documented 1-minute gap. False-alarm exposure is
the union duration of label-defined interictal EEG; preictal, ictal, and
ambiguous windows are not counted in that denominator.

The four values are SOP durations (5, 10, 15, and 30 minutes), not four
different SPHs. The current experiment fixes SPH at 5 minutes. For each
patient/model/horizon fold, the validation-only threshold is selected to
maximize event sensitivity subject to the requested false-alarm budget; the
held-out test patient is used only once for final metrics. Window-level and
event-level sensitivity are reported as separate outcomes.

Outputs in `artifacts/results-multihorizon-16case/` include per-patient/horizon/false-alarm-budget results,
aggregate means/standard deviations, fold-specific checkpoints, selected
validation thresholds, and an evaluation manifest. Report event sensitivity,
false alarms/hour, and lead time separately from window-level sensitivity/F1;
they are different measures.

For paper-oriented uncertainty analysis, regenerate unique-person mean estimates
with 10,000 person-cluster bootstrap draws and paired comparisons against the
matched baselines and ablations:

```powershell
python scripts\paired_bootstrap_analysis.py --results-dir artifacts\results-multihorizon-16case --draws 10000 --seed 20261016
```

The script writes `person_bootstrap_confidence_intervals.csv`,
`paired_model_comparisons.csv`, and `statistical_uncertainty_report.md` beside
the LOSO results. The bootstrap unit is the unique person (including combining
chb01/chb21), not EEG windows. These exploratory intervals do not correct for
multiple comparisons and are uncertain with only 15 people.

Use `--resume` to continue an interrupted run in the same output directory. It
retains complete patient/model result groups, reconstructs missing metrics from
compatible saved checkpoints, and trains only folds without a valid checkpoint.
It also reconstructs missing explanation audits from completed primary-model
checkpoints. Checkpoint fold assignments and experiment settings are validated
before reuse. To force CPU execution in PowerShell, set
`$env:CUDA_VISIBLE_DEVICES="-1"` before the evaluation command; otherwise CUDA
is used automatically when available.

If all checkpoints are present but a patient-results CSV was interrupted or
damaged, rebuild its metrics without retraining using
`python scripts\rebuild_multihorizon_results.py`. The rebuild validates each
checkpoint's fold metadata and atomically replaces the patient and summary CSVs.

The optional `--samples-per-epoch` cap preserves class-/patient-balanced
weighted sampling while bounding runtime; omit it to sample every labeled
training window each epoch. For a quick software smoke test, pass
`--max-recordings-per-patient 1` to `prepare` and `--max-folds 2 --epochs 1` to
`evaluate`. Such a run is not evidence of scientific performance. Full LOSO is
computationally expensive.

After LOSO model selection is complete, train a final inference checkpoint using
all included patients. This post-evaluation fit has no held-out test cohort and
must not be used to claim independent performance. Patient/state-balanced
sampling limits memory use; by default it draws 6,000 windows per epoch. Adjust
`--samples-per-epoch` for available memory and runtime:

```powershell
python -m seizure_prediction.cli fit-multihorizon-all --cache-dir artifacts\cache --checkpoint artifacts\results-multihorizon-16case\final_model\model.pt --model multi_horizon --epochs 3 --samples-per-epoch 6000
```

The checkpoint records its training patients, EDF count, available window
count, sampled windows per epoch, and protocol. It is the default checkpoint
used by the local API.

For explainability with SHAP in addition to the built-in Integrated Gradients and attention views, install the optional dependency with `python -m pip install -e ".[explain]"`.

The preictal SOP duration is swept across 5, 10, 15, and 30 minutes by default. The seizure prediction horizon (SPH) is 5 minutes: preictal windows fall in `[onset − SOP − SPH, onset − SPH)`. Windows within the configurable interictal exclusion buffer are discarded rather than mislabeled. These are explicit starting choices because the PDF does not specify exact window length, exclusion duration, or all preprocessing parameters; edit `ExperimentConfig` to document and reproduce approved alternatives.

Outputs include per-fold and aggregate CSV metrics, model checkpoints, and JSON run configuration. The final held-out patient's EEG is not used to train or select that fold's model. Every eligible patient is held out once. Within each fold, a distinct patient is reserved for validation.

## React + FastAPI interface

The bundled React build is served directly by the local FastAPI service. Start the complete application with:

```powershell
python run_project.py
```

For frontend development with Vite hot reload, install Node.js, run `npm ci`
in `frontend/` with npm's cache set to an E: project directory, then start with
`python run_project.py --dev`.

The app exposes the CHB-MIT prediction pipeline through `/api/health`, `/api/system-info`, `/api/overview`, `/api/reports`, and `/api/analyze-edf`. It loads saved evaluation summaries and accepts EDF uploads for inference using `artifacts/results-multihorizon-16case/final_model/model.pt`.

EDF uploads are limited to 256 MB, written temporarily under `tmp_uploads/`,
and deleted after analysis. Preprocessing uses the model's canonical montage;
unmatched montage channels are zero-filled and reported. The response includes
the available 5-, 10-, 15-, and 30-minute SOP scores, per-window predictions,
and fusion attention. Select the SOP horizon and score threshold for the live
window labels. The default 0.5 threshold is not a validation-calibrated alarm
threshold and provides no false-alarm guarantee on new patients. Integrated
Gradients displays channel, time, and time-frequency attribution for the
highest-scoring window. SHAP is optional and can be enabled after installing
`python -m pip install -e ".[explain]"`. These are model attributions, not a
validated seizure focus.

The results view distinguishes the validation false-alarm target from the
observed held-out false-alarm rate and reports event sensitivity separately
from window-level F1. The report view displays the full generated result report.
The fixed SPH is 5 minutes. All scores are retrospective research outputs, not
calibrated clinical probabilities or warnings.

To rebuild the bundled React assets after changing the frontend:

```powershell
Set-Location frontend
$env:npm_config_cache = "$PWD\.npm-cache"
npm ci
npm run build
```

`artifacts/results-multihorizon-16case/multihorizon_manifest.json` records the LOSO
model, alarm, horizon, and threshold-selection protocols.

The same inference path is available from the command line:

```powershell
python -m seizure_prediction.cli predict --edf path\to\recording.edf --checkpoint path\to\model.pt --output prediction.csv
```

## Results and artifacts

The completed experiment uses 16 CHB-MIT case folders (15 distinct people)
and writes its trained checkpoints, evaluation CSVs, and report under
`artifacts/results-multihorizon-16case/`. The completed experiment contains
1,344 case/horizon/alarm-budget rows (7 model variants × 16 held-out cases × 4
horizons × 3 budgets), 1,260 unique-person/horizon/alarm-budget rows (7 × 15
people × 4 × 3), and a 372-row event explanation audit. The
`multihorizon_loso_person_summary.csv` file reports mean ± SD across distinct
people, combining chb01 and chb21. The older full backup is `deliverables/Old_Project_Backup.zip`; it predates the latest Siena evaluation and file-name cleanup. The current E-drive project folder is authoritative. `deliverables/Project_Guide_and_Siena_Results.zip` contains the Siena report/results update. A new full ZIP can be generated with the package command below when E: has enough spare room for a second roughly 12-GB archive. Regenerable `artifacts/cache/` arrays can be recreated with the `prepare` command above. Because alarm thresholds
are selected on validation patients, the held-out test false-alarm rate can
exceed its target; the final report shows the observed rate and how many people
met each target.

Recreate the unique-person tables from the detailed held-out results with:

```powershell
python scripts\summarize_patient_level_results.py --results-dir artifacts\results-multihorizon-16case
```

Rebuild the full ZIP (saved on E:) after source or result changes with:

```powershell
python scripts\package_project.py
```

The React and FastAPI app uses the extracted
`artifacts/results-multihorizon-16case` directory for model inference and evaluation
summaries. Reinstall dependencies in a fresh Python environment using Quick start; Node is only
needed when changing and rebuilding the frontend.
the original machine's virtual environment and external CUDA installation are
not included in the archive.

The earlier single-horizon 30-minute run is retained only as historical output.
The later data audit found a URL-escaped annotated recording and filename-order
timeline errors, so its mean window sensitivity 0.590, specificity 0.556, F1
0.272, and PR-AUC 0.323 are **not a valid comparator** to the corrected
multi-horizon run. The separately mentioned 14.3% sensitivity has no documented
metric definition and must not be compared with either result.

## Important interpretation limits

- The CHB-MIT cohort is pediatric and does not establish performance on other populations or acquisition systems.
- A high window-level score is not by itself an operational seizure warning. The retrospective alarm protocol reports event-level sensitivity, false alarms per interictal hour, and lead time separately from window-level metrics.
- Siena external validation is reported separately and is not included in CHB-MIT LOSO claims.
- Attention and input attributions show model associations, not a clinically validated seizure focus.
- Literature comparisons in the PDF are illustrative and explicitly require final citations; no fabricated citations or results are included here.


For the short guide-friendly explanation and project folder map, open `Project_Explanation.md`. To launch the dashboard, double-click `Run_Project.bat` or run `python run_project.py`.
