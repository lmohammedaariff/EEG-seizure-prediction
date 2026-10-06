# Proposal coverage register

This register audits the supplied 21-page signed proposal against this software project. Proposal statements are treated as research targets unless the proposal gives an already-reported preliminary value. No target contribution is represented as an established result.

| Proposal page | Content to preserve | Project status |
|---|---|---|
| 1 | M.Tech AI project: patient-independent seizure prediction with multi-domain EEG; CHB-MIT; prediction plus state classification | Project title and scope retained in README and app. |
| 2 | Motivation: cross-patient variability, imbalance, channel/recording differences | Addressed by subject-held-out LOSO, capped/weighted training, channel standardization, and reports. |
| 3 | Compare domain adaptation, domain generalization, and ViT/unsupervised adaptation; replace illustrative entries with exact literature citations | Comparative-literature table was explicitly marked illustrative in the proposal. The repository must not present its example study labels as verified citations; populate `literature_review_template.csv` with the finalized, checked sources before thesis submission. |
| 4 | Research gaps: patient-specific learning, window leakage, imbalance, single domain, explainability, weak unseen-patient proof | Addressed by patient-level folds, multi-domain input, class weighting, explanation utilities, and patient-wise reporting. |
| 5 | Primary preictal vs non-preictal prediction; supporting interictal/preictal/ictal classification | Separate model heads, prediction CSVs and three-state CSVs; ictal examples are excluded from binary negatives. |
| 6 | Four objectives: standardized data, multi-domain learning, subject invariance/LOSO, explainability/prototype | Implemented in the research package; see remaining runtime/data prerequisites below. |
| 7 | CHB-MIT primary data; optional Siena external data if montage compatible | CHB-MIT ingestion and an opt-in Siena loader/evaluator are implemented. Siena's official seizure list is parsed; EKG channels are ignored; the 10-20 scalp channels are converted to the CHB bipolar montage and incompatible EDFs fail instead of silently padding. |
| 8 | End-to-end EDF → preprocessing → labeling → domains → attention → subject invariance → prediction/classification → explanation/interface | Pipeline and Streamlit prediction/results interface implemented. |
| 9 | Explicit preictal interval, interictal exclusion, ambiguous windows, SPH sweep at 5/10/15/30 minutes | Configurable horizons and onset/interictal/postictal buffers; transition-crossing windows excluded and reason-coded. Optional separate interictal/postictal buffer sweeps write excluded-window counts and patient metrics. |
| 10 | Temporal CNN, PSD/band power, STFT/time-frequency CNN | Three input branches implemented. |
| 11 | Attention/gated fusion and domain weights | Attention fusion and per-window attention values implemented; mean-fusion stages serve as ablations. |
| 12 | Shared state extractor plus adversarial patient classifier/gradient reversal | Implemented for the complete model stage. |
| 13 | Central three-domain → attention → invariance → explainability → both task outputs | Implemented. |
| 14 | LOSO, each eligible held-out patient, no window leakage, mean ± SD; optional Siena | LOSO and patient-wise reports implemented. CHB patient-clock labels span adjacent EDFs when summary clock starts are available. Siena validation is available by explicit opt-in after the dataset is placed at `data/siena`. |
| 15 | CNN, LSTM, CNN-LSTM baselines and staged ablation under the same LOSO | Implemented with common folds, training settings, and frozen threshold. |
| 16 | Preliminary CNN-LSTM history: train 73.2%, 73.4%, 74.6%; validation 46.7%, 48.2%, 45.5% at epochs 9–11; approximate final gap 29 points | These are proposal-reported preliminary figures, not reproduced by this project. They should be shown only with that attribution and must not be combined with new LOSO results. |
| 17 | Attention, Integrated Gradients, SHAP, channel/time/time-frequency importance; no clinical focus claim | Attention is plotted; the app can calculate Integrated Gradients and optional SHAP for the highest-scoring input window and display channel, band, temporal, and time-frequency attribution. All explanations are model-input influence only. |
| 18 | Target contributions explicitly unproven | README states hypotheses only; no superiority/SOTA/clinical-validity claim. |
| 19 | Roadmap: existing baseline, fixed protocol, proposed model, LOSO, ablations, explanations, optional Siena, paper/thesis | Reproducible software phases documented. No original baseline repository was supplied in this workspace, so its CNN/LSTM/CNN-LSTM comparisons were reimplemented; the original deployed application could not be migrated. Literature references, external data, experiments, and thesis/paper writing still require research work. |
| 20 | Final narrative: baseline gap motivates proposed approach; LOSO and ablation evaluate it | No conclusions are claimed before experiment results. |
| 21 | Supervisor/department/date sign-off page | Administrative signature page, not a model requirement; retained in the supplied source PDF. |

## Material still dependent on external inputs

1. CHB-MIT EDF files and the exact release used in the thesis.
2. Finalized literature-survey citations (page 3 labels were illustrative).
3. If Siena is used: the data download, 10-20 to CHB bipolar montage compatibility, and manual verification of summary time conversion against each EDF.
4. Real LOSO runs; the supplied preliminary accuracy chart is not a substitute for the required sensitivity, specificity, F1, PR-AUC, and patient-wise mean ± SD.
5. Clinical validation would be required before making any clinical-focus or deployment claim.
