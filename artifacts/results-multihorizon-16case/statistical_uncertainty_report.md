# Statistical uncertainty and paired model comparisons

Results use 15 unique held-out people. The `chb01`/`chb21` cases are combined before resampling.
Bootstrap intervals resample people with replacement (not windows or seizures); paired model contrasts use the same resampled people for both models. Intervals are percentile 95% confidence intervals with a fixed seed. These exploratory intervals do not correct for multiple comparisons and are imprecise with this small cohort.

A paired interval entirely above zero favors the shared multi-horizon model for event sensitivity, while for false alarms/hour a value below zero favors the shared model. Overlap with zero does not establish equivalence.

## Shared multi-horizon model at SOP 30 minutes

| Validation FAR target (/h) | Metric | Mean | Person-bootstrap 95% CI | n people |
|---:|---|---:|---:|---:|
| 0.1 | event_sensitivity | 0.006 | [0.000, 0.017] | 15 |
| 0.1 | false_alarms_per_hour | 0.018 | [0.000, 0.055] | 15 |
| 0.5 | event_sensitivity | 0.783 | [0.600, 0.933] | 15 |
| 0.5 | false_alarms_per_hour | 0.810 | [0.284, 1.620] | 15 |
| 1 | event_sensitivity | 0.918 | [0.840, 0.981] | 15 |
| 1 | false_alarms_per_hour | 0.571 | [0.346, 0.870] | 15 |

## Paired SOP 30-minute model contrasts

The table reports shared multi-horizon minus comparator using the same people for both models. Direction counts are descriptive: for sensitivity, better means higher; for FAR, better means lower. They do not replace the paired bootstrap interval or establish statistical significance.

| FAR target (/h) | Comparator | Metric | Mean paired difference | Person-bootstrap 95% CI | Reference better / tie / comparator better (people) |
|---:|---|---|---:|---:|---:|
| 0.5 | multi_horizon_cnn | event_sensitivity | -0.017 | [-0.050, 0.000] | 0 / 14 / 1 |
| 0.5 | multi_horizon_cnn | false_alarms_per_hour | -0.130 | [-0.380, 0.124] | 4 / 9 / 2 |
| 0.5 | multi_horizon_cnn_lstm | event_sensitivity | 0.028 | [0.000, 0.083] | 1 / 14 / 0 |
| 0.5 | multi_horizon_cnn_lstm | false_alarms_per_hour | 0.305 | [-0.027, 0.806] | 3 / 9 / 3 |
| 0.5 | multi_horizon_lstm | event_sensitivity | 0.050 | [0.000, 0.150] | 1 / 14 / 0 |
| 0.5 | multi_horizon_lstm | false_alarms_per_hour | 0.516 | [0.000, 1.397] | 0 / 12 / 3 |
| 0.5 | multi_horizon_multi_domain | event_sensitivity | 0.006 | [0.000, 0.017] | 1 / 14 / 0 |
| 0.5 | multi_horizon_multi_domain | false_alarms_per_hour | 0.053 | [-0.144, 0.337] | 4 / 8 / 3 |
| 0.5 | multi_horizon_temporal | event_sensitivity | 0.033 | [-0.050, 0.150] | 1 / 13 / 1 |
| 0.5 | multi_horizon_temporal | false_alarms_per_hour | -0.399 | [-2.096, 1.000] | 3 / 10 / 2 |
| 0.5 | multi_horizon_temporal_spectral | event_sensitivity | -0.033 | [-0.083, 0.000] | 0 / 13 / 2 |
| 0.5 | multi_horizon_temporal_spectral | false_alarms_per_hour | -1.028 | [-2.598, -0.059] | 7 / 8 / 0 |

Full interval and comparison tables are `person_bootstrap_confidence_intervals.csv` and `paired_model_comparisons.csv`. Thresholds remain those selected on the fold-specific validation person; the test participant is never used for threshold selection.
