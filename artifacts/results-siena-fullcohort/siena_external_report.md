# Siena external evaluation

This report evaluates frozen CHB-MIT LOSO checkpoints on the complete locally downloaded Siena cohort. Each checkpoint uses its own fold-specific threshold selected on that fold's separate CHB-MIT validation patient; Siena labels were not used to tune thresholds.

Cohort: 14 participants, 41 EDF recordings, and 47 annotated seizure events. Models: multi_horizon, multi_horizon_cnn, multi_horizon_lstm, multi_horizon_cnn_lstm. Horizons: 5, 10, 15, and 30 minutes (fixed 5-minute SPH).

## Main external result

At 30-minute SOP, using the CHB-validation threshold selected for a 0.1 false-alarm/hour budget, the proposed multi-horizon model detected 0/47 Siena events (0% event sensitivity) with zero false alarms. At the threshold selected for a 0.5/hour budget, it detected 45/47 events (95.7%) but produced 9.13 false alarms per eligible interictal hour. This is a marked failure to transfer the CHB operating point. Thresholds for nominal 0.5 and 1.0/hour budgets can coincide when validation selects the same operating threshold.


Subjects: PN00, PN01, PN03, PN05, PN06, PN07, PN09, PN10, PN11, PN12, PN13, PN14, PN16, PN17
Checkpoint fold: `chb08` (held-out from CHB-MIT fitting).

## Pooled results

| model_variant | horizon_minutes | far_budget_per_hour_chb_validation | subjects | events | eligible_events | detected_events | pooled_event_sensitivity | mean_subject_event_sensitivity | false_alarms | eligible_interictal_hours | pooled_false_alarms_per_hour | mean_subject_false_alarms_per_hour | mean_lead_time_minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi_horizon | 5 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 70.24027777777779 | 0.0 | 0.0 |  |
| multi_horizon | 5 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 87 | 70.24027777777779 | 1.238605580052597 | 1.3252760118710338 | 9.80177304964539 |
| multi_horizon | 5 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 87 | 70.24027777777779 | 1.238605580052597 | 1.3252760118710338 | 9.80177304964539 |
| multi_horizon | 10 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 67.48888888888891 | 0.0 | 0.0 |  |
| multi_horizon | 10 | 0.5 | 14 | 47 | 47 | 39 | 0.8297872340425532 | 0.8511904761904762 | 544 | 67.48888888888891 | 8.060586104708591 | 8.84676782246247 | 13.556837606837608 |
| multi_horizon | 10 | 1.0 | 14 | 47 | 47 | 39 | 0.8297872340425532 | 0.8511904761904762 | 544 | 67.48888888888891 | 8.060586104708591 | 8.84676782246247 | 13.556837606837608 |
| multi_horizon | 15 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 64.80416666666666 | 0.0 | 0.0 |  |
| multi_horizon | 15 | 0.5 | 14 | 47 | 47 | 36 | 0.7659574468085106 | 0.7583333333333334 | 501 | 64.80416666666666 | 7.73098437600463 | 8.571194350164445 | 15.775925925925925 |
| multi_horizon | 15 | 1.0 | 14 | 47 | 47 | 36 | 0.7659574468085106 | 0.7583333333333334 | 501 | 64.80416666666666 | 7.73098437600463 | 8.571194350164445 | 15.775925925925925 |
| multi_horizon | 30 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 57.15694444444445 | 0.0 | 0.0 |  |
| multi_horizon | 30 | 0.5 | 14 | 47 | 47 | 45 | 0.9574468085106383 | 0.9464285714285714 | 522 | 57.15694444444445 | 9.132748523801423 | 13.489331846830705 | 29.75555555555556 |
| multi_horizon | 30 | 1.0 | 14 | 47 | 47 | 45 | 0.9574468085106383 | 0.9464285714285714 | 522 | 57.15694444444445 | 9.132748523801423 | 13.489331846830705 | 29.75555555555556 |
| multi_horizon_cnn | 5 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 70.24027777777779 | 0.0 | 0.0 |  |
| multi_horizon_cnn | 5 | 0.5 | 14 | 47 | 47 | 45 | 0.9574468085106383 | 0.9404761904761905 | 591 | 70.24027777777779 | 8.413975836909021 | 10.514528642176247 | 9.274444444444443 |
| multi_horizon_cnn | 5 | 1.0 | 14 | 47 | 47 | 45 | 0.9574468085106383 | 0.9404761904761905 | 591 | 70.24027777777779 | 8.413975836909021 | 10.514528642176247 | 9.274444444444443 |
| multi_horizon_cnn | 10 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 1 | 67.48888888888891 | 0.014817253868949618 | 0.003913277424193831 |  |
| multi_horizon_cnn | 10 | 0.5 | 14 | 47 | 47 | 46 | 0.9787234042553191 | 0.9642857142857143 | 574 | 67.48888888888891 | 8.50510372077708 | 9.56590900391173 | 14.294565217391304 |
| multi_horizon_cnn | 10 | 1.0 | 14 | 47 | 47 | 46 | 0.9787234042553191 | 0.9642857142857143 | 574 | 67.48888888888891 | 8.50510372077708 | 9.56590900391173 | 14.294565217391304 |
| multi_horizon_cnn | 15 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 64.80416666666666 | 0.0 | 0.0 |  |
| multi_horizon_cnn | 15 | 0.5 | 14 | 47 | 47 | 46 | 0.9787234042553191 | 0.9642857142857143 | 387 | 64.80416666666666 | 5.971838230566451 | 7.984269714781969 | 18.718478260869563 |
| multi_horizon_cnn | 15 | 1.0 | 14 | 47 | 47 | 46 | 0.9787234042553191 | 0.9642857142857143 | 387 | 64.80416666666666 | 5.971838230566451 | 7.984269714781969 | 18.718478260869563 |
| multi_horizon_cnn | 30 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 57.15694444444445 | 0.0 | 0.0 |  |
| multi_horizon_cnn | 30 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 239 | 57.15694444444445 | 4.181469151702184 | 6.365802115534913 | 32.21631205673759 |
| multi_horizon_cnn | 30 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 239 | 57.15694444444445 | 4.181469151702184 | 6.365802115534913 | 32.21631205673759 |
| multi_horizon_lstm | 5 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 70.24027777777779 | 0.0 | 0.0 |  |
| multi_horizon_lstm | 5 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 43 | 70.24027777777779 | 0.612184367152433 | 0.9824741366986698 | 9.771631205673758 |
| multi_horizon_lstm | 5 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 43 | 70.24027777777779 | 0.612184367152433 | 0.9824741366986698 | 9.771631205673758 |
| multi_horizon_lstm | 10 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 67.48888888888891 | 0.0 | 0.0 |  |
| multi_horizon_lstm | 10 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 46 | 67.48888888888891 | 0.6815936779716824 | 1.0947851968433875 | 14.649290780141843 |
| multi_horizon_lstm | 10 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 46 | 67.48888888888891 | 0.6815936779716824 | 1.0947851968433875 | 14.649290780141843 |
| multi_horizon_lstm | 15 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 5 | 64.80416666666666 | 0.07715553269465698 | 0.11477506520333683 |  |
| multi_horizon_lstm | 15 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 41 | 64.80416666666666 | 0.6326753680961873 | 1.1146565674865594 | 19.25035460992908 |
| multi_horizon_lstm | 15 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 41 | 64.80416666666666 | 0.6326753680961873 | 1.1146565674865594 | 19.25035460992908 |
| multi_horizon_lstm | 30 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 2 | 57.15694444444445 | 0.03499137365441159 | 0.08000459240349825 |  |
| multi_horizon_lstm | 30 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 40 | 57.15694444444445 | 0.6998274730882317 | 1.598517239859335 | 32.06418439716312 |
| multi_horizon_lstm | 30 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 40 | 57.15694444444445 | 0.6998274730882317 | 1.598517239859335 | 32.06418439716312 |
| multi_horizon_cnn_lstm | 5 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 70.24027777777779 | 0.0 | 0.0 |  |
| multi_horizon_cnn_lstm | 5 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 47 | 70.24027777777779 | 0.669131750143357 | 1.028970528491996 | 9.730851063829789 |
| multi_horizon_cnn_lstm | 5 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 47 | 70.24027777777779 | 0.669131750143357 | 1.028970528491996 | 9.730851063829789 |
| multi_horizon_cnn_lstm | 10 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 67.48888888888891 | 0.0 | 0.0 |  |
| multi_horizon_cnn_lstm | 10 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 43 | 67.48888888888891 | 0.6371419163648335 | 1.0935167284354568 | 14.645744680851065 |
| multi_horizon_cnn_lstm | 10 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 43 | 67.48888888888891 | 0.6371419163648335 | 1.0935167284354568 | 14.645744680851065 |
| multi_horizon_cnn_lstm | 15 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 64.80416666666666 | 0.0 | 0.0 |  |
| multi_horizon_cnn_lstm | 15 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 40 | 64.80416666666666 | 0.6172442615572559 | 1.1062865183125203 | 19.257446808510636 |
| multi_horizon_cnn_lstm | 15 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 40 | 64.80416666666666 | 0.6172442615572559 | 1.1062865183125203 | 19.257446808510636 |
| multi_horizon_cnn_lstm | 30 | 0.1 | 14 | 47 | 47 | 0 | 0.0 | 0.0 | 0 | 57.15694444444445 | 0.0 | 0.0 |  |
| multi_horizon_cnn_lstm | 30 | 0.5 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 38 | 57.15694444444445 | 0.66483609943382 | 1.590415928459552 | 32.08723404255319 |
| multi_horizon_cnn_lstm | 30 | 1.0 | 14 | 47 | 47 | 47 | 1.0 | 1.0 | 38 | 57.15694444444445 | 0.66483609943382 | 1.590415928459552 | 32.08723404255319 |

## Cross-dataset interpretation: measured findings and hypotheses

**Measured and supported by this experiment:** the selected CHB-MIT validation thresholds do not preserve the target FAR on Siena. At 30-minute SOP, the full model changes from 0/47 detected events at 0.1/hour to 45/47 at the 0.5/hour CHB target, while observed FAR rises to 9.13/hour. The external evaluation used one frozen `chb08` fold per model, so fold-to-fold transfer variability was not estimated. Siena labels were not used to choose these thresholds.

**Dataset differences supported by official metadata and implemented preprocessing:** CHB-MIT is a pediatric cohort with 256-Hz EEG and predominantly bipolar derivations; Siena contains 14 adult participants and 512-Hz referential 10-20 EEG channels. The pipeline resamples both datasets to 128 Hz and constructs the canonical bipolar montage from Siena's referential electrodes. These acquisition and population differences are present; their causal contribution to the observed performance drop has not been isolated. Sources: [CHB-MIT PhysioNet documentation](https://physionet.org/content/chbmit/1.0.0/) and [Siena PhysioNet documentation](https://physionet.org/content/siena-scalp-eeg/1.0.0/).

**Plausible but untested explanations:** residual distribution shift from recording hardware, reference and channel availability; differences in participant age or seizure characteristics; and a probability-calibration shift under the CHB-derived threshold. The observed FAR increase is evidence of failed operating-point transfer, but it does not show whether score ranking, probability calibration, montage conversion, or cohort composition is the main cause. Testing these explanations requires per-fold transfer, channel-coverage/domain-shift analysis, and separately labeled calibration experiments; Siena must remain untouched for the primary external test.

## Limitations

This is a full evaluation of the 14-participant Siena cohort downloaded from PhysioNet, not clinical validation. The external test uses one frozen CHB-MIT chb08 LOSO checkpoint per model; it does not measure average transfer across all CHB folds. The database's recording protocol and reference montage differ from CHB-MIT; Siena recordings were converted to the project's fixed bipolar montage. Siena labels were not used for threshold tuning. Event sensitivity and false alarms per hour must be read together with eligible seizure counts and interictal exposure. A per-subject false-alarm rate is undefined when a participant contributes no eligible interictal exposure.

## Data citation

Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/5d4a-j060