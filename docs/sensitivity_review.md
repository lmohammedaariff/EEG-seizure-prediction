# Sensitivity and false-alarm review

## Protocol

The primary endpoint is event-level seizure sensitivity. False alarms are
merged when their alert windows are within one minute, and the denominator is
eligible interictal recording exposure. SOP values are 5, 10, 15, and 30
minutes with a fixed 5-minute SPH. Thresholds are selected on the separate
validation person only, maximizing event sensitivity under the specified FAR
budget. The test person is not used for threshold selection.

The validation protocol was extended with an alert-persistence analysis: an
alarm is emitted only after two consecutive 10-second windows with at least
50% overlap exceed the threshold. This requires a 15-second span of sustained
evidence. Thresholds are recalibrated on the validation participant with that
same rule, then applied to held-out cases. The saved checkpoints are reused;
the network is not retrained for this analysis.

For uncertainty reporting, a person-cluster bootstrap was added to the primary
one-window result table. Each draw samples the 15 unique people with replacement
and retains the paired model identity within each sampled person. This avoids
pretending that overlapping EEG windows or multiple seizures are independent
participants. Percentile 95% intervals and paired differences are in
`artifacts/results-multihorizon-16case/`; they are exploratory, are not adjusted
for multiple comparisons, and cannot overcome the small cohort size.

## Held-out CHB-MIT result at a 30-minute SOP

Mean ± sample SD across 15 unique people; chb01/chb21 are combined.

| Model | Validation FAR budget (/h) | One-window sensitivity | One-window FAR (/h) | 2-window sensitivity | 2-window FAR (/h) | People meeting FAR: one → two |
|---|---:|---:|---:|---:|---:|---:|
| Multi-horizon | 0.1 | 0.006 ± 0.022 | 0.018 ± 0.071 | 0.012 ± 0.047 | 0.101 ± 0.392 | 14 → 14 |
| Multi-horizon | 0.5 | 0.783 ± 0.352 | 0.810 ± 1.440 | 0.789 ± 0.352 | 0.821 ± 1.451 | 10 → 10 |
| Multi-horizon | 1.0 | 0.918 ± 0.147 | 0.571 ± 0.539 | 0.918 ± 0.147 | 0.643 ± 0.590 | 13 → 12 |
| LSTM baseline | 0.1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.009 | 15 → 15 |
| LSTM baseline | 0.5 | 0.733 ± 0.406 | 0.294 ± 0.260 | 0.756 ± 0.371 | 0.430 ± 0.420 | 12 → 11 |
| LSTM baseline | 1.0 | 0.918 ± 0.147 | 0.403 ± 0.215 | 0.868 ± 0.278 | 0.370 ± 0.273 | 15 → 15 |

Persistence produced small sensitivity changes, not a consistent improvement
in the sensitivity/FAR trade-off. The strict 0.1/h setting remains unusable
for useful event sensitivity. At the 0.5/h target, the full multi-horizon
model's observed mean FAR is above the target and highly variable across
people. The persistence gate therefore does not establish dependable
false-alarm control. All horizons, person-level measurements, and fold-level
thresholds are in `artifacts/results-multihorizon-16case/` files ending in
`persistence2.csv`.

## External Siena result

The complete downloaded cohort included 14 participants, 41 EDF recordings,
47 annotated seizures, and 57.16 eligible interictal hours for the 30-minute
SOP. Each frozen architecture used its `chb08` checkpoint and CHB validation
threshold; Siena labels were not used to tune the operating point.

| Model | CHB validation FAR target (/h) | Events detected | Event sensitivity | Observed FAR (/eligible interictal h) |
|---|---:|---:|---:|---:|
| Multi-horizon | 0.1 | 0/47 | 0.0% | 0.00 |
| Multi-horizon | 0.5 | 45/47 | 95.7% | 9.13 |
| CNN | 0.5 | 47/47 | 100% | 4.18 |
| LSTM | 0.5 | 47/47 | 100% | 0.70 |
| CNN-LSTM | 0.5 | 47/47 | 100% | 0.66 |

For the proposed multi-horizon model, strict 0.1/h thresholds detected 0/47
events at every horizon. At 0.5/h, event detection was 47/47, 39/47, 36/47,
and 45/47 at 5/10/15/30-minute SOPs, while pooled observed FAR was
1.24/8.06/7.73/9.13 per eligible hour. Some budgets map to the same threshold
when validation selects the same operating point. The data are seizure-rich
and provide only about 57 eligible interictal hours, which limits FAR
precision. The external results show that CHB-calibrated thresholds fail to
transfer reliably; they do not identify a deployment-ready threshold.

This is a full evaluation of the locally downloaded Siena cohort, but it uses
one frozen CHB fold (`chb08`) per architecture rather than averaging transfer
across all CHB folds. It is not prospective or clinical validation. Full
per-subject and all-horizon results are in
`artifacts/results-siena-fullcohort/siena_external_report.md`.

## Interpretation

Class imbalance handling, correct patient grouping, and threshold selection
on a separate validation subject are implemented and audited, but the model's
operating point remains a limitation. A better result must be demonstrated
on untouched people at the event level while reporting FAR and lead time.
Neither the CHB-MIT results nor the Siena external evaluation support a
clinical warning claim.
