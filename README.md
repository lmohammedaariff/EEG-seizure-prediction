# Patient-independent EEG seizure prediction

This is an executable research prototype implementing the supplied thesis scope: CHB-MIT preprocessing, traceable interval labels, multi-domain EEG, attention fusion, subject-adversarial learning, two distinct prediction/classification heads, LOSO folds, patient-wise metrics, ablations, and a Streamlit inference/results interface. Proposed improvements are hypotheses; this project does not claim they are effective until experiments are run.

The proposal's research hypotheses are tracked separately in [`docs/target_hypotheses.csv`](docs/target_hypotheses.csv). Its page 3 related-work examples were explicitly illustrative, so [`docs/proposal_related_work_examples.csv`](docs/proposal_related_work_examples.csv) preserves those comparison axes without inventing citations; fill the verified study details from the finalized literature survey before thesis submission. [`docs/proposal_coverage.md`](docs/proposal_coverage.md) maps every page of the supplied proposal to the implementation or names remaining external inputs.

## Install

Use Python 3.10 or newer. From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[explain]"
```

Run the explicit label-boundary checks with the standard-library test runner:

```powershell
python -m unittest discover -s tests -v
```

Place the extracted CHB-MIT patient folders and their `*-summary.txt` files under `data/chbmit/`, for example `data/chbmit/chb01/`. The proposal describes 664 EDF recordings at 256 Hz and approximately 42.6 GB. The official release summary is used for seizure onset/end annotation parsing; keep filenames and patient folders intact. See [the official CHB-MIT release](https://physionet.org/content/chbmit/1.0.0/).

## Train and evaluate

Fast first run (still performs a full LOSO pass for one horizon):

```powershell
seizure-prediction --config config.yaml --quick
```

Full proposal experiment (CNN, LSTM, CNN-LSTM baselines, then all ablation stages at 5, 10, 15, and 30 minutes):

```powershell
seizure-prediction --config config.yaml
```

Choose an explicit subset when iterating:

```powershell
seizure-prediction --config config.yaml --horizons 5 15 --stages temporal_only temporal_spectral multi_domain multi_domain_attention complete
```

To measure ambiguous-window sensitivity without conflating buffer types, sweep the interictal and postictal buffers separately at the values in `config.yaml`:

```powershell
seizure-prediction --config config.yaml --quick --exclusion-sensitivity
```

This writes separate runs and patient-wise scores (including excluded-window counts) under `artifacts/exclusion_sensitivity/`.

Optional external validation on Siena (after running the desired CHB-MIT stages and horizons):

```powershell
seizure-prediction --config config.yaml --horizons 5 10 15 30 --stages complete --external-siena
```

Put Siena patient folders and their `Seizures-list-PNxx.txt` files under `data/siena/`. The loader reads registration/seizure clock times, converts them to recording-relative seconds, ignores EKG, resamples 512 Hz input, and derives the CHB bipolar montage from 10-20 electrode channels. By default it stops if any required derivation is missing. If you manually approve zero-filling absent montage channels for an explicitly documented sensitivity run, set `siena_allow_missing_channels: true`; missing derivations are recorded in the window audit CSV. Manually verify clock conversion and channel montage against the EDFs before reporting external results. Siena is optional and does not replace CHB-MIT LOSO. The official dataset describes 14 patient folders and 512 Hz recordings with seizure lists beside the EDFs; see [the official Siena release](https://physionet.org/content/siena-scalp-eeg/1.0.0/).

Each held-out patient is excluded from all training windows for that fold. Prediction threshold is fixed in advance at `0.5` by default; no held-out patient is used for threshold tuning. Patient-wise prediction and state-classification files, checkpoints, and mean/SD summaries are written under `artifacts/`. Window caps in the configuration make a laptop-scale run possible and use evenly spaced windows per EDF and state; set them to `0` to retain all windows (may require substantial RAM and compute).

## Run the interface

```powershell
streamlit run streamlit_app.py
```

The app displays LOSO and optional Siena reports when available and accepts an EDF plus a fold checkpoint to generate per-window preictal scores, three-state predictions, and attention weights. It can calculate Integrated Gradients and optional SHAP for the highest-scoring window, then visualize domain, channel, band, time, and time-frequency importance. These explanations describe model input influence, not clinical focus localization; prediction scores are research outputs, not clinical alarms.

## Operational definitions

- Each prediction is made from one configurable EEG input window (`window_seconds`, default 4 seconds; `overlap`, default 50%). This is separate from the preictal horizon.
- For onset `T` and configured horizon `H`, preictal candidate windows lie in `[T-H, T)`. Where CHB-MIT summary file start clocks are available, the labeler builds a patient-level clock timeline so windows in the prior EDF can be preictal for an upcoming seizure. If those clocks are missing, it falls back to recording-relative labels and marks the time reference. A window fully within an annotated seizure interval is ictal. Any window that straddles interval edges or falls in configured exclusion buffers is labeled `-1` and excluded from fitting and scoring, never reassigned to interictal.
- The proposal draws an SOP boundary but supplies no separate numeric SOP duration. This implementation keeps the configured preictal horizon, prediction-window length, seizure onset, and named exclusion buffers separate; it does not invent another SOP setting.
- The onset, preictal-to-interictal, and postictal exclusion buffers are independent configurable seconds in `config.yaml`. Defaults are zero seconds at preictal boundaries beyond excluding windows that cross interval boundaries, and five minutes after ictal offset. Record these choices with each experiment.
- The primary task is preictal versus non-preictal. Ictal windows are not treated as negative prediction examples. The separate supporting head classifies interictal, preictal, and ictal states.
- The explicit interval logic lives in `src/labeling/exclusion.py`. Every retained sample carries patient/recording/window identifiers, start/end seconds, seizure identifiers and boundaries when applicable, horizon, exclusion setting, label, and label reason. `max_windows_per_recording_per_class` downsamples valid samples evenly per recording; boundary-excluded windows retain their exclusion reason.
- The CLI writes a separate `window_labels_{H}m.csv` audit table for every candidate window, including `included_in_dataset` so a downsampled window can be distinguished from one used to fit/evaluate a model.
- Preprocessing uses resampling to 256 Hz, a configurable 0.5–40 Hz Butterworth bandpass, configurable 60 Hz notch, causal filtering, and robust per-window channel scaling (so a window does not use statistics from later in its recording). EDF channel order is standardized to common CHB-MIT bipolar channels, with zero padding if a channel is absent. Review channel names and filtering choices as implementation decisions before freezing thesis experiments.

## Models and reporting

`cnn`, `lstm`, and `cnn_lstm` use only temporal raw EEG. Ablations then progress through temporal-only, temporal+spectral, full multi-domain mean fusion, multi-domain attention fusion, and the complete model with gradient-reversal subject classification. Every model uses the same held-out patient folds. The state head is trained with all three labels; the prediction loss is applied only to non-ictal examples. Square-root positive weighting is capped at 20 to reduce class-imbalance effects without using held-out data.

Primary per-patient metrics are sensitivity, specificity, F1, and average precision (the reported stepwise PR-AUC); each report includes mean and standard deviation across eligible patients. Accuracy is not used as the primary success criterion. The separate state task also reports accuracy and macro-F1. Sparse folds with no positives have undefined PR-AUC and are preserved as `NaN` in reports.

The proposal's preliminary CNN-LSTM figures (epochs 9–11, training 73.2/73.4/74.6% and validation 46.7/48.2/45.5%) are transcribed into the app with a clear source note. They are not reproduced here and are not LOSO evidence.

## Explainability

The model exposes attention weights across active domains. `seizure_prediction.explain` provides Integrated Gradients for temporal, spectral, and time-frequency inputs, aggregated channel/time/frequency importance, and optional SHAP GradientExplainer. Install `.[explain]` for Captum and SHAP. Explanations show model input influence only and do not identify a clinical seizure focus.

## Limits and reproducibility notes

- This workspace did not include CHB-MIT EDF data, so no training results are included or claimed.
- The supplied proposal requests per-patient LOSO evaluation. This implementation uses fixed epochs and a predeclared 0.5 decision threshold; tuning epochs, loss weights, or thresholds requires a training-only nested/inner patient validation protocol.
- CHB-MIT annotations are parsed from each patient's summary in recording-relative seconds, then projected to the patient clock when file start times are present. Verify summary parsing, recording naming, clock rollover, and montage consistency against the downloaded release before treating outputs as thesis results.
- Siena remains optional; CHB-MIT LOSO is the primary evaluation. Verify Siena summary timing and the 10-20-to-CHB bipolar conversion against each EDF before interpreting external results.
- The all-data audit CSV records each sliding candidate window and whether it was retained under the configured per-recording cap; model arrays are capped to make the full research pipeline executable on practical hardware.
- Siena clock-time conversion and CHB-MIT patient-clock reconstruction should be visually reviewed against the original summary files; ambiguous or missing clocks fall back to recording-relative labeling rather than guessed continuity.
- For a publication-quality experiment, save the exact config, dependency lock, data release/version, random seed, and commit alongside each output directory.
