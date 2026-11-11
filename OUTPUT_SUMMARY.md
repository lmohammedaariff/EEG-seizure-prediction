# Output summary for review

## CHB-MIT patient-held-out experiment

- 16 included case folders representing 15 distinct people; chb01 and chb21 are grouped as one person.
- Prediction horizons: 5, 10, 15, and 30 minutes, with a fixed 5-minute seizure prediction horizon.
- Seven evaluated variants: four ablation stages and matched CNN, LSTM, and CNN-LSTM baselines.
- Event-level results use validation-calibrated thresholds targeting 0.1, 0.5, and 1.0 false alarms per hour. Held-out false-alarm rates can exceed the validation target.
- At 30-minute SOP and the 0.5/hour validation target, the full model achieved 78.3% ± 35.2% mean event sensitivity and 0.81 ± 1.44 observed false alarms/hour across 15 people; 10/15 people met the target.
- 98 of 100 annotated CHB-MIT seizures were evaluable under the current labeling and recording-coverage rule.

## Siena external transfer

- 14 people and 47 annotated seizures; one frozen CHB-MIT chb08 fold was evaluated per model.
- For the full model at 30-minute SOP, CHB-derived thresholds detected 0/47 events at the 0.1/hour target and 45/47 at the 0.5/hour target, with 9.13 observed false alarms/hour.
- This is retrospective external-dataset evaluation, not clinical validation. Possible causes of transfer failure remain hypotheses unless separately tested.

## Detailed outputs

- `artifacts/results-multihorizon-16case/final_results_report.md`: model/horizon/FAR comparison and patient-wise summaries.
- `artifacts/results-multihorizon-16case/statistical_uncertainty_report.md`: person-cluster confidence intervals and paired comparisons.
- `docs/evaluation_integrity_audit.md`: saved-fold split, labels, threshold and metric arithmetic audit.
- `artifacts/results-siena-fullcohort/siena_external_report.md`: Siena results, measured findings and hypotheses.

Research use only; not for medical care.
