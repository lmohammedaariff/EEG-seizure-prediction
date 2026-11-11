# LOSO evaluation integrity audit

This audit checks the saved CHB-MIT result tables against the run manifest and reviews the preprocessing, labeling, split, threshold, and alarm code paths. It is a reproducibility/code audit; it is not an independent clinical annotation audit.

## Automated artifact checks

- Fold-level table: 1,344 rows; expected grid: 7 model variants × 16 held-out case folders × 4 SOPs × 3 FAR targets = 1,344 rows.
- Unique-person table: 15 people after combining chb01/chb21; 1,260 model/person/SOP/FAR rows.
- Split audit: PASS; each distinct test person and validation person is excluded from that fold’s training cases.
- Metric arithmetic: event sensitivity equals detected events divided by eligible events; FAR equals false-alarm count divided by eligible interictal exposure hours.
- Label audit: 64 case-by-SOP rows.

| SOP (min) | People | Annotated events | Evaluable events | Detected by full model at 0.5/h validation target |
|---:|---:|---:|---:|---:|
| 5 | 15 | 100 | 98 | 79/98 |
| 10 | 15 | 100 | 98 | 78/98 |
| 15 | 15 | 100 | 98 | 77/98 |
| 30 | 15 | 100 | 98 | 74/98 |

The two annotated events without an evaluable forecast interval are excluded by the event eligibility rule because no fully labeled positive forecast window is available under the current recording coverage. Event-level inference uses 15 people as the independent unit; the 98 eligible seizures are not treated as 98 independent subjects.

## Protocol checks from source code and manifests

- **Preprocessing:** 0.5–45 Hz band-pass, 60 Hz notch where below Nyquist, resampling to 128 Hz, 10-second windows, and 50% overlap. Each window is normalized channel-wise. Missing canonical channels are zero-filled and recorded in preprocessing manifests.
- **Label construction:** SOP-specific preictal interval is `[onset − SOP − SPH, onset − SPH)`; SPH is 5 minutes. Windows must lie wholly inside the forecast interval to receive a preictal label. A 60-minute interictal exclusion buffer is applied; ictal and ambiguous windows are not used as binary negatives.
- **Annotations and timelines:** CHB-MIT summary onset/offset times are reconciled to EDF recordings. Absolute EDF header start timestamps are preferred for recording offsets; source summary clocks are the fallback. Summary-referenced missing EDFs and raw/cache mismatches cause errors.
- **Patient separation:** chb01 and chb21 share one person group. The splitter excludes both case folders from training/validation whenever either case is held out. A distinct person supplies validation; the held-out person supplies test results.
- **Threshold selection:** validation-only code calibrates each model/SOP threshold to maximize event sensitivity under the selected FAR target. The saved manifest records an 81-point probability grid plus a no-alarm sentinel; test metrics do not select thresholds.
- **Event scoring:** alarm scores become available at the end of each window; alarms within a 1-minute merge gap are combined. An event is detected by a unique alarm onset in its SOP/SPH forecast interval. FAR exposure is the union duration of label-defined interictal EEG; preictal, ictal, and ambiguous time are excluded.

## Scope limits and items to retain in the paper

- This result set covers 16 CHB-MIT case folders (15 people), not the complete database. Siena transfer uses one frozen CHB-MIT chb08 fold per architecture; it is not an average across all CHB folds.
- Siena referential channels are converted to the project’s canonical bipolar montage. This is a documented preprocessing decision and a possible source of residual domain shift, not proof of the cause of the transfer failure.
- The independent participant count is 15. Bootstrap intervals and paired comparisons are exploratory and are not corrected for multiple comparisons.
- Source-code review found no train/validation/test patient overlap in the saved folds. Automated pytest checks were not run in this environment because pytest is unavailable in the project and bundled Python environments.

## Audit result

All saved-grid, split, denominator, and metric arithmetic checks passed.
