# Thesis proposal implementation and evidence checklist

This checklist maps the signed proposal to the delivered software and the
evidence currently available. It distinguishes completed implementation from
claims that require more data or validation.

| Proposal item | Delivered implementation/evidence | Status and limits |
|---|---|---|
| Patient-independent epileptic seizure prediction using multi-domain EEG | Shared multi-horizon model, four SOP heads, three-state output, and React + FastAPI interface | Implemented and trained on 16 included CHB-MIT cases (15 distinct people). This is a research system, not a clinical warning device. |
| Inter-patient variability and subject leakage | LOSO evaluation; validation patient separated from test patient; chb01/chb21 same-person split guard | Implemented for all 16 available case folders; chb01 and chb21 are grouped as one person in every train/validation/test split. |
| Imbalanced Interictal / Preictal / Ictal labels | SOP-specific labels, boundary exclusion, balanced patient/state sampling, class-weighted horizon loss, label audit | Implemented; see `artifacts/results-multihorizon-16case/label_audit.csv` and experiment manifest. |
| Related-work comparison and research gap | DOI-linked literature review with a bounded contribution statement; adjacent 2026 cross-subject ViT/domain-adaptation seizure-detection work is distinguished from forecasting | References and distinctions are listed in `docs/research_gap.md`. No claim of being the first multi-horizon, LOSO, attention, adversarial, or explainable seizure model. |
| Three-state classification and prediction distinction | Separate state-classification and preictal-vs-interictal prediction outputs; fixed 5-minute SPH | Implemented. The four swept values are SOPs of 5, 10, 15, and 30 minutes. |
| CHB-MIT dataset | Sixteen downloaded case folders and their EDFs/summaries are stored under E: and included in the rebuilt ZIP | Included subset only. The full PhysioNet database has more cases and recordings; results must not be described as full-dataset LOSO. |
| Siena external generalization | Full downloaded cohort: 14 participants, 41 EDFs, 47 annotated seizures. Frozen CHB-MIT `chb08` fold, CHB validation-only thresholds, four models × four horizons; Siena referential channels converted to the fixed bipolar montage | Full downloaded cohort evaluation completed. At SOP 30m, proposed model: 0/47 detected at 0.1/h threshold; 45/47 at 0.5/h with 9.13 observed false alarms per eligible interictal hour. LSTM: 47/47 at 0.5/h with 0.70 FAR/h; CNN-LSTM: 47/47 with 0.66 FAR/h. This is one-fold transfer evidence, not mean transfer across CHB folds or clinical validation. See `artifacts/results-siena-fullcohort/`. |
| Preprocessing and windowing | EDF parsing, 60-Hz notch, 0.5–45-Hz filtering, 128-Hz resampling, canonical bipolar montage, fixed 10-second windows with 50% overlap | Implemented. Missing montage channels are zero-filled and reported. |
| Temporal, spectral, and time-frequency representations | Temporal CNN, PSD/band-power branch, STFT time-frequency CNN | Implemented in the full multi-domain model. |
| Attention/gated fusion | Learned attention over the three feature domains | Implemented; weights can be shown with live explanations. |
| Subject-invariant representation | Gradient-reversal subject adversary trained with seizure-related heads | Implemented in the full model and included in the ablation grid. |
| LOSO and patient-wise reporting | Held-out case folds with separate validation patient; case-level results plus unique-person aggregation combining chb01/chb21 | Completed for 16 held-out case folders representing 15 people. The person-level table contains 1,260 person/model/horizon/FAR-budget rows (7 variants × 15 people × 4 SOPs × 3 budgets); the detailed case table contains 1,344 rows. |
| CNN, LSTM, CNN-LSTM comparisons | Matched multi-head baselines evaluated on the same folds | Completed across the 16 available case folds. Results are in `artifacts/results-multihorizon-16case/`. |
| Ablation sequence | Temporal; temporal+spectral; multi-domain attention; full subject-invariant model | Completed across the 16 available case folds with matched LOSO splits. |
| Sensitivity, false alarms, lead time, and window diagnostics | Person-level event sensitivity, observed false alarms per eligible interictal hour, lead time, window sensitivity/F1/PR-AUC; two-consecutive-window alert persistence compared with one-window alerts using validation-only thresholds | Reported separately with mean ± SD across 15 people. Persistence gave only small, inconsistent changes; it did not solve the sensitivity/FAR trade-off. Siena shows threshold-transfer failure. See `docs/sensitivity_review.md` and result files ending in `persistence2.csv`. |
| Statistical uncertainty and baseline contrasts | Person-cluster bootstrap confidence intervals and paired comparisons for event sensitivity and false alarms/hour across all SOPs and validation FAR targets | Completed with 10,000 bootstrap draws from 15 unique people. Exploratory intervals are reported; multiple comparisons are not corrected and intervals do not establish equivalence. See `artifacts/results-multihorizon-16case/statistical_uncertainty_report.md`. |
| Explainability | Attention weights, Integrated Gradients channel/time/time-frequency attribution, optional GradientSHAP channel attribution; stability/perturbation audit outputs | Implemented. The 16-case audit covers 93 seizure events per horizon (372 event/horizon explanation rows). SHAP requires the optional `.[explain]` install. Explanations describe model behavior, not a clinically validated seizure focus. |
| Streamlit deployment shown in original proposal | Replaced by a bundled React UI served by FastAPI | Includes saved LOSO sensitivity/FAR plots; local CHB-MIT EDF selection or upload; recording-wide risk curve; real multichannel trace, relative spectrum, and STFT view. The normal launcher requires Python only; Node is needed only to develop/rebuild the frontend. |
| Preliminary CNN-LSTM accuracy plot and generalization gap | Kept as historical proposal context, not relabeled as new experiment results | The proposal's 74.6%/45.5% epoch chart was not reproduced as an independent LOSO result. The corrected study reports its own patient-held-out metrics. |
| Thesis/paper readiness | Reproducible commands, manifest, literature review, person-level report, architecture diagram, traceability checklist, current CHB-MIT code/model/results/data ZIP, and full downloaded-cohort Siena evaluation artifacts | Siena external evaluation is complete for the downloaded cohort. Remaining research limits: CHB-MIT covers only 16 case folders/15 unique people, transfer tests one frozen CHB fold, sensitivity/FAR trade-offs remain, and no prospective/clinical validation exists. The updated full ZIP is `deliverables/SeizurePredictionResearch-complete.zip`; the Siena data and results are included except raw EDFs, whose checksum manifest is included. |

## Reproduce the reported experiment

From the project root, create the Python environment and keep temporary files
and package cache under the E: project directory, as shown in the main README.
The included processed-window cache is excluded from the ZIP and can be
regenerated from the included EDFs. Rebuilding the full experiment is
computationally expensive; use the saved CSVs/report to inspect the completed
run before retraining.

## Main evidence files

- `artifacts/results-multihorizon-16case/final_results_report.md`
- `artifacts/results-multihorizon-16case/multihorizon_loso_per_patient.csv`
- `artifacts/results-multihorizon-16case/multihorizon_loso_summary.csv`
- `artifacts/results-multihorizon-16case/multihorizon_loso_per_person.csv`
- `artifacts/results-multihorizon-16case/multihorizon_loso_person_summary.csv`
- `artifacts/results-multihorizon-16case/multihorizon_manifest.json`
- `artifacts/results-multihorizon-16case/label_audit.csv`
- `artifacts/results-multihorizon-16case/explanation_audit_summary.csv`
- `artifacts/results-multihorizon-16case/person_bootstrap_confidence_intervals.csv`
- `artifacts/results-multihorizon-16case/paired_model_comparisons.csv`
- `artifacts/results-multihorizon-16case/statistical_uncertainty_report.md`
- `docs/architecture.svg`
- `docs/research_gap.md`
