# Literature review and research gap

## Scope and claim discipline

This is a focused, narrative review of multi-horizon seizure forecasting,
patient-independent evaluation, alarm-level metrics, and explanation evaluation.
It is not a systematic review and cannot support a "first" claim. Several prior
studies already use forecast intervals, cross-patient evaluation, and
interpretable models. The contribution proposed here is a **combined experimental
question and evaluation protocol**, not a claim that any one component is new.

## Relevant prior work

| Work | Evidence relevant to this project | What it does not establish for this project |
|---|---|---|
| Cheng & Goldenholz (2025), [Seizure prediction and forecasting: a scoping review](https://doi.org/10.1097/WCO.0000000000001344) | Reviews current forecasting translation challenges, including external validation and benchmark choice; cautions that seizure frequency affects reported metrics and discusses a moving-average benchmark. | A scoping review does not establish one universally accepted forecasting protocol. |
| Chang et al. (2024), [Necessary for seizure forecasting outcome metrics: Seizure frequency and benchmark model](https://doi.org/10.1016/j.eplepsyres.2024.107474) | Shows that seizure frequency affects many evaluation metrics and that moving-average forecasts can be a stronger comparator than permutation-only chance baselines. | The evaluated diaries, wearables, and simulations are not the same task or recording protocol as this CHB-MIT scalp-EEG subset. |
| Shafiezadeh et al. (2024), [Calibrating Deep Learning Classifiers for Patient-Independent Electroencephalogram Seizure Forecasting](https://doi.org/10.3390/s24092863) | Directly studies patient-independent EEG forecasting and calibration, establishing that cross-patient prediction is prior art. Their calibration condition uses seizure information from the new patient. | Calibration with a held-out patient's seizure is not zero-shot LOSO generalization; report it separately if ever added. |
| Yang et al. (2024), [Seizure forecasting with ultra long-term EEG signals](https://doi.org/10.1016/j.clinph.2024.09.017) | Evaluates hourly and daily forecasts on ultra-long-term implanted/subcutaneous recordings and includes cross-patient analyses. | Those devices, forecast timescales, and cohorts differ from short-horizon CHB-MIT scalp EEG. It rules out broad claims that patient-independent forecasting is unexplored. |
| Kim et al. (2023), [Exploration of Short-range Neonatal Seizure Forecasting with Quantitative EEG Based Deep Learning](https://doi.org/10.1101/2023.08.01.23293285) (preprint) | Studies multiple seizure-prediction horizons/occurrence periods and demonstrates that prior work already varies forecast intervals. | Neonatal forecasting with a different split and EEG population is not directly comparable to held-out pediatric scalp EEG in this experiment. |
| Skoric et al. (2026), [Forecasting and real-time detection of neonatal seizures: A machine learning perspective](https://doi.org/10.1016/j.cmpb.2026.109552) | Reports multiple forecast intervals for neonatal EEG; further evidence against claiming that "multi-horizon seizure forecasting" alone is novel. | Neonatal short-range prediction does not answer this CHB-MIT patient-held-out task. |
| Gao et al. (2024), [An interpretable and generalizable deep learning model for iEEG-based seizure prediction using prototype learning and contrastive learning](https://doi.org/10.1016/j.compbiomed.2024.109257) | Combines cross-patient iEEG prediction and prototype-based interpretation. | Intracranial EEG prototypes differ from scalp EEG input attributions; explainability alone is not novel, and prototype similarity is not physiological validation. |
| Feng et al. (2026), [Cross-subject seizure detection with vision transformer and unsupervised domain adaptation](https://doi.org/10.1016/j.bspc.2025.108341) | Adjacent work applies a Vision Transformer and unsupervised domain adaptation to cross-subject EEG seizure **detection**, showing that transformer/domain-adaptation ideas are also prior art. | Detection labels ongoing seizure activity; it does not establish preictal forecasting at multiple SOPs. It is therefore not a direct forecasting comparator. |
| Kuhlmann et al. (2018), [Seizure prediction—ready for a new era](https://doi.org/10.1038/s41582-018-0055-2) | Foundational review of seizure-prediction definitions and evaluation challenges. | It predates recent benchmarking and model work; use it as context, not the sole state-of-the-art survey. |

## Specific research gap and contribution hypothesis

The individual elements—multiple forecast intervals, LOSO-style patient
separation, CNN/LSTM baselines, attention, adversarial domain learning, and
attribution methods—are established ideas. The research question is whether, on
this **specific limited CHB-MIT scalp-EEG cohort**, a shared multi-domain
representation with separate SOP-specific forecast heads can achieve a useful
event-sensitivity/false-alarm trade-off on patients excluded from both training
and threshold selection, whether sequential temporal-only → temporal+spectral →
multi-domain attention → subject-adversarial ablations improve that trade-off
over matched CNN/LSTM baselines, and whether its EEG explanations are stable and
perturbation-faithful.

**Proposed contribution statement (conditional on reported experiments):**

> We conduct a patient-held-out comparison of sequential multi-horizon
> representation ablations and a shared multi-domain EEG forecaster against
> matched multi-head CNN, LSTM, and CNN-LSTM baselines. We select alarm
> thresholds only on a distinct validation patient to
> maximize event sensitivity under prespecified false-alarm-per-hour budgets,
> then report event sensitivity, false alarms per eligible interictal hour,
> prediction lead time, and window metrics on the untouched test patient. We
> audit channel attributions for repeatability across preictal windows and
> faithfulness under channel perturbation.

This is a **study/protocol contribution hypothesis**, not a "first-ever model"
claim. It is supported only if the matched experiments and explanation audits
are actually run and results disclosed. Sixteen case folders representing 15
people from one public, pediatric dataset cannot establish external/site
generalization or clinical utility.

## Operational definitions used in this implementation

- SOP labels are generated independently for each requested horizon. With the
  configured SPH, a positive window falls in
  `[onset - SOP - SPH, onset - SPH)`; only windows wholly inside that interval
  are preictal. Ictal windows and ambiguous boundary/exclusion windows are not
  binary negatives.
- Summary-referenced EDFs are checked against both the raw recordings and
  prepared cache. Absolute recording offsets are derived from EDF header
  timestamps when present, so recordings are not assumed to be chronological
  in filename order.
- The multi-horizon model shares its signal representation and emits one
  preictal probability head per SOP. Head-specific labels are masked when that
  window is excluded for that SOP.
- Each LOSO test patient is held out from training and model selection. A
  different patient is used for validation, and the remaining patients train
  the fold. No held-out test labels select the alarm threshold.
- CHB-MIT case `chb21` is a later recording from the same person as `chb01`.
  The split helper groups these case IDs so one cannot enter training or
  validation when the other is held out. Both case folders are included. The
  raw LOSO table has 16 held-out case rows but 15 unique people; the person-level
  summary combines `chb01` and `chb21` before calculating means and SDs.
- Alarm thresholds are separately selected on validation predictions at
  0.1, 0.5, and 1.0 false alarms/hour. The event forecast interval and the
  alarm-merging gap are recorded in the run manifest.
- A threshold-crossing window becomes observable when the window ends;
  overlapping threshold crossings are merged into alert episodes. Event
  sensitivity counts annotated seizures with a unique alert onset in the
  event's SOP/SPH forecast interval; a single alert cannot detect two events.
  Lead time is seizure onset minus alert onset.
- False alarms/hour use the union duration of label-defined interictal EEG as
  exposure; preictal, ictal, and ambiguous windows are excluded. Report
  eligible events/exposure alongside rates because short recordings and rare
  events make the denominator consequential.
- A two-window persistence policy was evaluated separately from the model: the
  threshold must be exceeded on two consecutive valid windows, and the alarm is
  timestamped at the end of the second window. Thresholds are recalibrated
  using validation-patient predictions with the same persistence rule.
- Window-level sensitivity and event-level seizure sensitivity are separate
  outcomes. The prior 30-minute experiment reported window-level metrics only;
  it did not establish event sensitivity, false alarms/hour, or lead time.

## Sensitivity and class imbalance

Preictal windows are a minority in the label audit, and the ratio changes with
the SOP and patient. Training therefore uses patient/state-balanced sampling;
the multi-horizon heads also use horizon-specific inverse-frequency class
weights computed from training patients only. `label_audit.csv` records the
preictal/interictal/ictal counts and ratios for each patient and SOP. Validation
threshold calibration then maximizes **event-level** sensitivity at each
prespecified false-alarm budget. This changes the operating point; it does not
guarantee that test sensitivity will improve. The held-out patient never selects
its threshold. Window sensitivity remains a separate diagnostic metric.
The validation budget is only a threshold-selection target; it cannot guarantee
the same false-alarm rate on unseen people. In the completed run, each
model/horizon/budget setting met its FAR target in 8 to 15 of the 15 unique
people after combining the paired `chb01`/`chb21` case results. The person-level
summary lists compliance beside observed FAR and sensitivity so the
generalization gap remains visible.

The standalone 14.3% sensitivity mentioned in earlier materials has no
documented evaluation file or metric definition. It cannot be interpreted as
window-level or event-level sensitivity until traced to a reproducible result.

## Explanation evaluation

Displaying SHAP or Integrated Gradients is not sufficient to claim a meaningful
explanation. The implemented audit (run for the proposed shared model) is:
(1) quantify channel-ranking repeatability across multiple preictal windows of
each held-out seizure; (2) compare risk change after masking the top-attributed
channels against random-channel masks of the same size; and (3) report
channel/domain/time-frequency importance as model associations. Repeatability
and perturbation faithfulness are necessary checks, not proof of a causal
seizure focus or clinical physiological validity.

## Dataset, prior benchmark, and limitations

The corrected subset includes 16 complete CHB-MIT case folders representing 15
people: `chb01`, `chb02`, `chb03`, `chb05`, `chb08`, `chb11`, `chb13`,
`chb14`, `chb16`, `chb17`, `chb19`, `chb20`, `chb21`, `chb22`, `chb23`, and
`chb24`. The manifest records 463 EDF recordings and 100 annotated seizure
events reconciled to their summaries. The event explanation audit includes 93
eligible events at each of four horizons (372 event/horizon rows). The previous
single-horizon 30-minute
run had mean **window-level** sensitivity 0.590, specificity 0.556, F1 0.272,
and PR-AUC 0.323, but it predates discovery of a URL-escaped annotated EDF and
filename-order timeline errors. Treat those values as historical and
**invalid as a comparator** to the corrected run. The separately mentioned
14.3% figure has no documented metric definition in the supplied materials.

The subset is too small and patient-variable to demonstrate clinical utility.
Patient-wise folds within one CHB-MIT database are not external validation.
Alarm timing, eligible exposure, and event eligibility must be interpreted
alongside the code and `multihorizon_manifest.json`; do not equate this
retrospective alarm policy with a prospective bedside-warning system.

## Siena external generalization evaluation

The complete locally downloaded cohort has been evaluated: 14 participants,
41 EDF recordings, and 47 annotated seizure events. Dataset metadata, format,
and license are documented on the
[official PhysioNet Siena page](https://physionet.org/content/siena-scalp-eeg/1.0.0/).

The external model was frozen from the CHB-MIT LOSO `chb08` fold. For each
model, SOP, and false-alarm budget, it uses the threshold selected on the
separate CHB-MIT validation participant for that exact `chb08` fold; Siena
labels were not used to tune it. Siena's 512-Hz referential channels were
converted to the project's canonical bipolar montage and resampled using the
same preprocessing. One published seizure offset fell after its EDF ended;
that offset was clipped to the recording boundary and recorded in
`artifacts/results-siena-fullcohort/siena_annotation_audit.csv`.

At a 30-minute SOP, the proposed model detected 0/47 events at its threshold
selected for 0.1 false alarms/hour on CHB validation, with zero observed
false alarms; at the 0.5/hour CHB validation threshold, it detected 45/47
(95.7%) but produced 9.13 false alarms/hour over 57.16 eligible interictal
hours. The matched LSTM detected 47/47 at that nominal 0.5/hour threshold
(0.70 observed FAR/hour), and CNN-LSTM detected 47/47 (0.66 FAR/hour). The
CNN detected 47/47 but produced 4.18 FAR/hour. At horizons 5/10/15/30, the
proposed model's strict 0.1/h threshold detected 0/47 events at each horizon;
at 0.5/h it detected 47/47, 39/47, 36/47, and 45/47, with pooled observed
FARs 1.24, 8.06, 7.73, and 9.13 per eligible hour, respectively. These are
results on a seizure-enriched public cohort and only 57 hours of eligible
interictal exposure, so FAR estimates remain limited. They show clear
cross-dataset operating-threshold failure, not a solved sensitivity/FAR
trade-off.

The external test uses one frozen CHB-MIT `chb08` fold per architecture, not
an average across all CHB folds. The Siena labels were not used to select
thresholds. This is a full evaluation of the downloaded 14-participant
cohort, but it is not prospective or clinical validation.

A two-consecutive-window alert-persistence gate was also evaluated on saved
CHB-MIT LOSO folds for the full multi-horizon model and LSTM baseline. Its
effects were small and inconsistent, and it did not improve FAR compliance
overall. The comparison and mean ± SD patient-level values are in
[`docs/sensitivity_review.md`](sensitivity_review.md) and the persistence2
result tables. This is evidence against presenting persistence by itself as
the solution to the observed sensitivity/FAR gap.


## Verified references

Metadata and claims below were checked against the publisher or database
records on 14 October 2026. The Kim et al. item is a preprint and is labeled as
such; it is not presented as peer-reviewed evidence.

1. Cheng, J. C., & Goldenholz, D. M. (2025). Seizure prediction and
   forecasting: a scoping review. *Current Opinion in Neurology, 38*(2),
   135–139. https://doi.org/10.1097/WCO.0000000000001344
2. Chang, C.-Y., Zhang, B., Moss, R., Picard, R., Westover, M. B., & Goldenholz,
   D. M. (2024). Necessary for seizure forecasting outcome metrics: Seizure
   frequency and benchmark model. *Epilepsy Research, 208*, 107474.
   https://doi.org/10.1016/j.eplepsyres.2024.107474
3. Shafiezadeh, S., Duma, G. M., Mento, G., Danieli, A., Antoniazzi, L., Del
   Popolo Cristaldi, F., Bonanni, P., & Testolin, A. (2024). Calibrating deep
   learning classifiers for patient-independent electroencephalogram seizure
   forecasting. *Sensors, 24*(9), 2863. https://doi.org/10.3390/s24092863
4. Yang, H., Müller, J., Eberlein, M., et al. (2024). Seizure forecasting with
   ultra long-term EEG signals. *Clinical Neurophysiology, 167*, 211–220.
   https://doi.org/10.1016/j.clinph.2024.09.017
5. Kim, J., Glass, H. C., Amorim, E., Rao, V. R., & Bernardo, D. (2023).
   Exploration of short-range neonatal seizure forecasting with quantitative
   EEG based deep learning. *medRxiv* [preprint].
   https://doi.org/10.1101/2023.08.01.23293285
6. Skoric, M., et al. (2026). Forecasting and real-time detection of neonatal
   seizures: A machine learning perspective. *Computer Methods and Programs in
   Biomedicine, 285*, 109552. https://doi.org/10.1016/j.cmpb.2026.109552
7. Gao, Y., Liu, A., Cui, H., Qian, R., & Chen, X. (2024). An interpretable and
   generalizable deep learning model for iEEG-based seizure prediction using
   prototype learning and contrastive learning. *Computers in Biology and
   Medicine, 183*, 109257.
   https://doi.org/10.1016/j.compbiomed.2024.109257
8. Kuhlmann, L., Lehnertz, K., Richardson, M. P., Schelter, B., & Zaveri, H. P.
   (2018). Seizure prediction—ready for a new era. *Nature Reviews Neurology,
   14*(10), 618–630. https://doi.org/10.1038/s41582-018-0055-2
9. Feng, H., Wang, S., Lv, H., Nie, C., Feng, W., Peng, H., & Zhao, Y. (2026).
   Cross-subject seizure detection with vision transformer and unsupervised
   domain adaptation. *Biomedical Signal Processing and Control, 111*, 108341.
   https://doi.org/10.1016/j.bspc.2025.108341
10. Shoeb, A. H., & Guttag, J. V. (2010). Application of machine learning to
   epileptic seizure detection. *Proceedings of the 27th International
   Conference on Machine Learning*, Haifa, Israel. https://archive.physionet.org/pn6/chbmit/shoeb-icml-2010.pdf
11. Shoeb, A. H., et al. (2010). CHB-MIT Scalp EEG Database, version 1.0.0.
    PhysioNet. https://doi.org/10.13026/C2K01R

The source slide's other example rows (including domain-generalization and
Vision-Transformer entries) are treated as prompts for literature checking,
not as verified findings. Only the cross-subject detection paper above is
included here, and it is explicitly distinguished from seizure forecasting.
