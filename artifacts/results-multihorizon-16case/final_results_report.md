# Multi-horizon CHB-MIT LOSO results

## Study protocol

The report aggregates 16 included CHB-MIT case folders (15 distinct people): `chb01`, `chb02`, `chb03`, `chb05`, `chb08`, `chb11`, `chb13`, `chb14`, `chb16`, `chb17`, `chb19`, `chb20`, `chb21`, `chb22`, `chb23`, `chb24`. Each test case is excluded from fitting and threshold selection; a separate participant group is reserved for validation. Cases `chb01` and `chb21` are never split across train/validation/test and are combined for person-level summaries.

A shared multi-horizon model predicts SOPs of 5, 10, 15, and 30 minutes with a fixed 5-minute SPH. Thresholds are selected on validation data to maximize event sensitivity under 0.1, 0.5, and 1.0 false alarms per interictal hour. These budgets are validation targets; actual held-out test rates can differ. Reported event sensitivity, false alarms/hour, and lead time are event-level outcomes. Window sensitivity and PR-AUC are separate window-level diagnostics.

## Results

The detailed table below shows case-level means ± sample SD. A unique-person headline is added after the table; it combines the paired `chb01`/`chb21` cases before averaging across people. Valid n shows the denominator for event sensitivity, false alarms/hour, lead time, window sensitivity, and window PR-AUC. Event sensitivity and window sensitivity are percentages; lead time is minutes.

| Model | SOP (min) | FAR target (/h) | Valid n (event/FAR/lead/window sens/PR-AUC) | Test cases meeting FAR target | Event sensitivity (%) | FAR (/h) | Lead time (min) | Window sensitivity (%) | Window PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_horizon | 5 | 0.1 | 16/16/0/16/16 | 15/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.0 ± 0.1 |
| multi_horizon_cnn | 5 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 5 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.0 ± 0.1 |
| multi_horizon_lstm | 5 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.0 ± 0.1 |
| multi_horizon_multi_domain | 5 | 0.1 | 16/16/1/16/16 | 15/16 | 1.0% ± 4.2% | 0.1 ± 0.2 | 7.5 ± n/a | 0.0% ± 0.1% | 0.1 ± 0.1 |
| multi_horizon_temporal | 5 | 0.1 | 16/16/0/16/16 | 14/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_temporal_spectral | 5 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon | 5 | 0.5 | 16/16/14/16/16 | 11/16 | 87.5% ± 34.2% | 0.5 ± 0.5 | 9.8 ± 0.2 | 82.9% ± 33.7% | 0.0 ± 0.1 |
| multi_horizon_cnn | 5 | 0.5 | 16/16/15/16/16 | 9/16 | 91.1% ± 26.4% | 1.1 ± 1.5 | 9.7 ± 0.3 | 83.6% ± 29.0% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 5 | 0.5 | 16/16/14/16/16 | 13/16 | 87.5% ± 34.2% | 1.1 ± 2.9 | 9.6 ± 0.3 | 62.1% ± 30.8% | 0.0 ± 0.1 |
| multi_horizon_lstm | 5 | 0.5 | 16/16/14/16/16 | 14/16 | 87.5% ± 34.2% | 0.4 ± 0.4 | 9.7 ± 0.2 | 57.0% ± 25.4% | 0.0 ± 0.1 |
| multi_horizon_multi_domain | 5 | 0.5 | 16/16/15/16/16 | 10/16 | 90.6% ± 27.2% | 1.3 ± 2.3 | 9.7 ± 0.2 | 82.5% ± 34.0% | 0.1 ± 0.1 |
| multi_horizon_temporal | 5 | 0.5 | 16/16/14/16/16 | 10/16 | 87.5% ± 34.2% | 1.6 ± 2.8 | 9.6 ± 0.5 | 75.5% ± 36.7% | 0.1 ± 0.1 |
| multi_horizon_temporal_spectral | 5 | 0.5 | 16/16/14/16/16 | 11/16 | 87.5% ± 34.2% | 1.6 ± 2.6 | 9.6 ± 0.3 | 75.8% ± 36.2% | 0.1 ± 0.1 |
| multi_horizon | 5 | 1 | 16/16/15/16/16 | 14/16 | 93.8% ± 25.0% | 0.5 ± 0.5 | 9.7 ± 0.2 | 89.2% ± 25.6% | 0.0 ± 0.1 |
| multi_horizon_cnn | 5 | 1 | 16/16/15/16/16 | 12/16 | 93.8% ± 25.0% | 0.7 ± 0.7 | 9.7 ± 0.2 | 88.3% ± 24.6% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 5 | 1 | 16/16/15/16/16 | 13/16 | 93.8% ± 25.0% | 1.3 ± 2.9 | 9.6 ± 0.3 | 66.0% ± 26.0% | 0.0 ± 0.1 |
| multi_horizon_lstm | 5 | 1 | 16/16/15/16/16 | 14/16 | 93.8% ± 25.0% | 0.4 ± 0.4 | 9.7 ± 0.2 | 61.0% ± 20.4% | 0.0 ± 0.1 |
| multi_horizon_multi_domain | 5 | 1 | 16/16/15/16/16 | 13/16 | 93.8% ± 25.0% | 1.0 ± 1.9 | 9.7 ± 0.2 | 88.0% ± 28.4% | 0.1 ± 0.1 |
| multi_horizon_temporal | 5 | 1 | 16/16/15/16/16 | 11/16 | 93.8% ± 25.0% | 1.6 ± 2.8 | 9.6 ± 0.5 | 81.7% ± 31.0% | 0.1 ± 0.1 |
| multi_horizon_temporal_spectral | 5 | 1 | 16/16/15/16/16 | 12/16 | 93.8% ± 25.0% | 1.6 ± 2.6 | 9.6 ± 0.3 | 82.1% ± 30.4% | 0.1 ± 0.1 |
| multi_horizon | 10 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_cnn | 10 | 0.1 | 16/16/1/16/16 | 15/16 | 1.2% ± 5.0% | 0.0 ± 0.1 | 9.4 ± n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 10 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_lstm | 10 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_multi_domain | 10 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_temporal | 10 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_temporal_spectral | 10 | 0.1 | 16/16/0/16/16 | 15/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon | 10 | 0.5 | 16/16/14/16/16 | 12/16 | 86.9% ± 34.0% | 0.5 ± 0.4 | 14.6 ± 0.5 | 83.0% ± 34.0% | 0.1 ± 0.1 |
| multi_horizon_cnn | 10 | 0.5 | 16/16/14/16/16 | 10/16 | 86.9% ± 34.0% | 0.9 ± 1.1 | 14.7 ± 0.3 | 82.0% ± 33.8% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 10 | 0.5 | 16/16/14/16/16 | 12/16 | 86.9% ± 34.0% | 1.0 ± 2.4 | 14.6 ± 0.4 | 64.2% ± 31.2% | 0.1 ± 0.1 |
| multi_horizon_lstm | 10 | 0.5 | 16/16/14/16/16 | 14/16 | 86.9% ± 34.0% | 0.4 ± 0.5 | 14.7 ± 0.3 | 54.7% ± 23.6% | 0.1 ± 0.1 |
| multi_horizon_multi_domain | 10 | 0.5 | 16/16/15/16/16 | 9/16 | 93.1% ± 25.0% | 1.0 ± 2.1 | 14.6 ± 0.5 | 83.8% ± 30.2% | 0.1 ± 0.1 |
| multi_horizon_temporal | 10 | 0.5 | 16/16/14/16/16 | 10/16 | 86.1% ± 33.8% | 1.6 ± 2.5 | 14.5 ± 0.6 | 74.0% ± 39.1% | 0.1 ± 0.1 |
| multi_horizon_temporal_spectral | 10 | 0.5 | 16/16/14/16/16 | 10/16 | 86.9% ± 34.0% | 2.1 ± 4.0 | 14.4 ± 0.9 | 72.9% ± 40.9% | 0.1 ± 0.1 |
| multi_horizon | 10 | 1 | 16/16/15/16/16 | 13/16 | 92.6% ± 24.9% | 0.5 ± 0.4 | 14.6 ± 0.5 | 89.2% ± 25.9% | 0.1 ± 0.1 |
| multi_horizon_cnn | 10 | 1 | 16/16/15/16/16 | 11/16 | 92.6% ± 24.9% | 0.9 ± 1.1 | 14.6 ± 0.4 | 88.3% ± 26.0% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 10 | 1 | 16/16/15/16/16 | 14/16 | 92.6% ± 24.9% | 1.0 ± 2.4 | 14.6 ± 0.4 | 69.2% ± 26.2% | 0.1 ± 0.1 |
| multi_horizon_lstm | 10 | 1 | 16/16/15/16/16 | 14/16 | 92.6% ± 24.9% | 0.4 ± 0.5 | 14.6 ± 0.4 | 59.3% ± 18.9% | 0.1 ± 0.1 |
| multi_horizon_multi_domain | 10 | 1 | 16/16/15/16/16 | 15/16 | 92.6% ± 24.9% | 0.5 ± 0.4 | 14.6 ± 0.4 | 88.5% ± 25.9% | 0.1 ± 0.1 |
| multi_horizon_temporal | 10 | 1 | 16/16/15/16/16 | 11/16 | 91.8% ± 24.8% | 1.6 ± 2.5 | 14.4 ± 0.6 | 80.2% ± 34.2% | 0.1 ± 0.1 |
| multi_horizon_temporal_spectral | 10 | 1 | 16/16/15/16/16 | 12/16 | 92.6% ± 24.9% | 2.1 ± 4.0 | 14.3 ± 0.9 | 79.1% ± 36.3% | 0.1 ± 0.1 |
| multi_horizon | 15 | 0.1 | 16/16/1/16/16 | 16/16 | 2.1% ± 8.3% | 0.0 ± 0.0 | 9.7 ± n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_cnn | 15 | 0.1 | 16/16/1/16/16 | 16/16 | 4.2% ± 16.7% | 0.0 ± 0.0 | 12.1 ± n/a | 0.1% ± 0.2% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 15 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_lstm | 15 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon_multi_domain | 15 | 0.1 | 16/16/1/16/16 | 15/16 | 0.9% ± 3.6% | 0.0 ± 0.1 | 17.2 ± n/a | 0.0% ± 0.0% | 0.2 ± 0.2 |
| multi_horizon_temporal | 15 | 0.1 | 16/16/0/16/16 | 15/16 | 0.0% ± 0.0% | 0.0 ± 0.1 | n/a | 0.0% ± 0.0% | 0.1 ± 0.2 |
| multi_horizon_temporal_spectral | 15 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.1 ± 0.1 |
| multi_horizon | 15 | 0.5 | 16/16/14/16/16 | 11/16 | 86.2% ± 34.0% | 1.4 ± 3.3 | 19.5 ± 0.6 | 76.5% ± 38.0% | 0.1 ± 0.1 |
| multi_horizon_cnn | 15 | 0.5 | 16/16/14/16/16 | 10/16 | 86.2% ± 34.0% | 1.1 ± 1.7 | 19.6 ± 0.5 | 79.3% ± 35.1% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 15 | 0.5 | 16/16/14/16/16 | 13/16 | 86.2% ± 34.0% | 0.5 ± 0.8 | 19.6 ± 0.5 | 64.3% ± 29.1% | 0.1 ± 0.1 |
| multi_horizon_lstm | 15 | 0.5 | 16/16/14/16/16 | 14/16 | 86.2% ± 34.0% | 0.4 ± 0.5 | 19.6 ± 0.5 | 57.9% ± 26.2% | 0.1 ± 0.1 |
| multi_horizon_multi_domain | 15 | 0.5 | 16/16/16/16/16 | 8/16 | 93.4% ± 21.7% | 1.0 ± 1.5 | 19.2 ± 1.4 | 86.2% ± 30.8% | 0.2 ± 0.2 |
| multi_horizon_temporal | 15 | 0.5 | 16/16/14/16/16 | 10/16 | 84.7% ± 33.9% | 1.4 ± 2.4 | 19.1 ± 1.8 | 73.9% ± 39.4% | 0.1 ± 0.2 |
| multi_horizon_temporal_spectral | 15 | 0.5 | 16/16/14/16/16 | 11/16 | 86.2% ± 34.0% | 2.1 ± 4.1 | 19.4 ± 0.9 | 73.2% ± 39.7% | 0.1 ± 0.1 |
| multi_horizon | 15 | 1 | 16/16/15/16/16 | 12/16 | 92.0% ± 25.1% | 1.5 ± 3.3 | 19.3 ± 1.0 | 82.6% ± 32.3% | 0.1 ± 0.1 |
| multi_horizon_cnn | 15 | 1 | 16/16/15/16/16 | 10/16 | 92.0% ± 25.1% | 1.1 ± 1.7 | 19.4 ± 1.0 | 85.6% ± 28.3% | 0.1 ± 0.1 |
| multi_horizon_cnn_lstm | 15 | 1 | 16/16/15/16/16 | 14/16 | 92.0% ± 25.1% | 0.6 ± 0.8 | 19.4 ± 1.0 | 68.7% ± 23.5% | 0.1 ± 0.1 |
| multi_horizon_lstm | 15 | 1 | 16/16/15/16/16 | 14/16 | 92.0% ± 25.1% | 0.4 ± 0.5 | 19.4 ± 1.0 | 62.4% ± 21.3% | 0.1 ± 0.1 |
| multi_horizon_multi_domain | 15 | 1 | 16/16/16/16/16 | 14/16 | 92.9% ± 21.6% | 0.6 ± 0.4 | 19.3 ± 1.1 | 91.1% ± 25.4% | 0.2 ± 0.2 |
| multi_horizon_temporal | 15 | 1 | 16/16/15/16/16 | 12/16 | 90.4% ± 25.3% | 1.4 ± 2.4 | 19.0 ± 1.9 | 80.1% ± 34.4% | 0.1 ± 0.2 |
| multi_horizon_temporal_spectral | 15 | 1 | 16/16/15/16/16 | 12/16 | 92.0% ± 25.1% | 2.1 ± 4.1 | 19.2 ± 1.2 | 79.3% ± 35.0% | 0.1 ± 0.1 |
| multi_horizon | 30 | 0.1 | 16/16/1/16/16 | 15/16 | 0.5% ± 2.1% | 0.0 ± 0.1 | 26.4 ± n/a | 0.0% ± 0.0% | 0.2 ± 0.2 |
| multi_horizon_cnn | 30 | 0.1 | 16/16/1/16/16 | 16/16 | 2.1% ± 8.3% | 0.0 ± 0.0 | 34.2 ± n/a | 0.0% ± 0.1% | 0.2 ± 0.2 |
| multi_horizon_cnn_lstm | 30 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.2 ± 0.2 |
| multi_horizon_lstm | 30 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.2 ± 0.2 |
| multi_horizon_multi_domain | 30 | 0.1 | 16/16/2/16/16 | 14/16 | 9.4% ± 25.8% | 2.8 ± 10.0 | 27.5 ± 2.0 | 1.3% ± 4.1% | 0.2 ± 0.2 |
| multi_horizon_temporal | 30 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.2 ± 0.2 |
| multi_horizon_temporal_spectral | 30 | 0.1 | 16/16/0/16/16 | 16/16 | 0.0% ± 0.0% | 0.0 ± 0.0 | n/a | 0.0% ± 0.0% | 0.2 ± 0.2 |
| multi_horizon | 30 | 0.5 | 16/16/14/16/16 | 11/16 | 79.7% ± 34.4% | 0.8 ± 1.4 | 32.9 ± 2.7 | 78.6% ± 38.7% | 0.2 ± 0.2 |
| multi_horizon_cnn | 30 | 0.5 | 16/16/14/16/16 | 9/16 | 81.2% ± 34.8% | 0.9 ± 1.6 | 32.8 ± 2.8 | 81.0% ± 36.3% | 0.2 ± 0.2 |
| multi_horizon_cnn_lstm | 30 | 0.5 | 16/16/14/16/16 | 12/16 | 77.1% ± 36.3% | 0.5 ± 0.7 | 32.1 ± 5.1 | 62.3% ± 32.4% | 0.2 ± 0.2 |
| multi_horizon_lstm | 30 | 0.5 | 16/16/13/16/16 | 13/16 | 75.0% ± 39.8% | 0.3 ± 0.3 | 33.3 ± 2.2 | 57.5% ± 30.0% | 0.2 ± 0.2 |
| multi_horizon_multi_domain | 30 | 0.5 | 16/16/14/16/16 | 10/16 | 79.2% ± 34.6% | 0.8 ± 1.3 | 32.8 ± 2.8 | 79.6% ± 37.4% | 0.2 ± 0.2 |
| multi_horizon_temporal | 30 | 0.5 | 16/16/13/16/16 | 11/16 | 76.6% ± 40.3% | 1.2 ± 2.7 | 33.0 ± 2.2 | 72.5% ± 42.6% | 0.2 ± 0.2 |
| multi_horizon_temporal_spectral | 30 | 0.5 | 16/16/14/16/16 | 10/16 | 82.8% ± 35.0% | 1.7 ± 3.3 | 32.6 ± 3.1 | 73.5% ± 39.2% | 0.2 ± 0.2 |
| multi_horizon | 30 | 1 | 16/16/16/16/16 | 14/16 | 92.3% ± 14.3% | 0.6 ± 0.5 | 32.8 ± 3.3 | 95.7% ± 11.1% | 0.2 ± 0.2 |
| multi_horizon_cnn | 30 | 1 | 16/16/16/16/16 | 13/16 | 92.3% ± 14.3% | 0.8 ± 0.8 | 32.8 ± 3.3 | 92.3% ± 12.9% | 0.2 ± 0.2 |
| multi_horizon_cnn_lstm | 30 | 1 | 16/16/16/16/16 | 15/16 | 92.3% ± 14.3% | 0.6 ± 0.9 | 32.7 ± 3.3 | 75.0% ± 15.9% | 0.2 ± 0.2 |
| multi_horizon_lstm | 30 | 1 | 16/16/16/16/16 | 16/16 | 92.3% ± 14.3% | 0.4 ± 0.2 | 32.7 ± 3.3 | 66.6% ± 13.0% | 0.2 ± 0.2 |
| multi_horizon_multi_domain | 30 | 1 | 16/16/16/16/16 | 14/16 | 93.9% ± 13.7% | 0.9 ± 1.8 | 32.5 ± 3.4 | 95.0% ± 13.4% | 0.2 ± 0.2 |
| multi_horizon_temporal | 30 | 1 | 16/16/16/16/16 | 12/16 | 93.9% ± 13.7% | 1.9 ± 3.3 | 32.5 ± 3.2 | 87.1% ± 25.8% | 0.2 ± 0.2 |
| multi_horizon_temporal_spectral | 30 | 1 | 16/16/16/16/16 | 13/16 | 93.9% ± 13.7% | 2.0 ± 3.6 | 32.4 ± 3.3 | 85.3% ± 28.2% | 0.2 ± 0.2 |

## Evaluable seizure denominators

The full multi-horizon model (`multi_horizon`) was evaluated on 100 annotated seizures across 15 unique people. Under the current SOP/SPH label and recording-coverage rule, 98 seizures had at least one fully labeled forecast window and were evaluable at each horizon; two annotated events lacked such a window. Event counts are not independent sample units; patient-level inference uses 15 people. The table reports pooled detections at the 0.5/hour validation-target operating point; observed held-out FAR may exceed that target.

| SOP (min) | Unique people | Annotated seizures | Evaluable seizures | Detected / evaluable at 0.5/h target |
|---:|---:|---:|---:|---:|
| 5 | 15 | 100 | 98 | 79/98 |
| 10 | 15 | 100 | 98 | 78/98 |
| 15 | 15 | 100 | 98 | 77/98 |
| 30 | 15 | 100 | 98 | 74/98 |

The 84-row comparison table above covers all seven variants (four ablations and three matched baselines), four SOPs, and three validation FAR targets. Fold-level metrics and per-fold evaluable/detected seizure counts are in [`multihorizon_loso_per_patient.csv`](multihorizon_loso_per_patient.csv); the unique-person table is in [`multihorizon_loso_per_person.csv`](multihorizon_loso_per_person.csv).

## Unique-person results

At the 30-minute SOP and 0.1/hour validation target, the full multi-domain, subject-adversarial model (`multi_horizon`) achieved event sensitivity **0.6% ± 2.2%**, observed FAR **0.0 ± 0.1/hour**, and mean warning lead **26.4 ± n/a minutes**; 14 of 15 people met the FAR target.
At the 30-minute SOP and 0.5/hour validation target, the full multi-domain, subject-adversarial model (`multi_horizon`) achieved event sensitivity **78.3% ± 35.2%**, observed FAR **0.8 ± 1.4/hour**, and mean warning lead **32.7 ± 2.7 minutes**; 10 of 15 people met the FAR target.
At the 30-minute SOP and 1/hour validation target, the full multi-domain, subject-adversarial model (`multi_horizon`) achieved event sensitivity **91.8% ± 14.7%**, observed FAR **0.6 ± 0.5/hour**, and mean warning lead **32.6 ± 3.3 minutes**; 13 of 15 people met the FAR target.

At the 30-minute SOP and 1.0/hour target, the matched LSTM baseline achieved event sensitivity **91.8% ± 14.7%** and observed FAR **0.4 ± 0.2/hour**, with 15 of 15 people meeting the target. The model variants trade sensitivity against false alarms; no single variant dominates every operating point.

## Label and imbalance audit

The label audit contains 64 patient-by-horizon rows. Preictal-to-interictal window ratios ranged from 0.0051 to 4.9618. Training uses patient/state-balanced sampling and horizon-specific loss weights estimated from training patients only. Ictal and ambiguous windows are excluded from the binary preictal-versus-interictal target.

## Explanation audit

Integrated Gradients channel rankings are evaluated for repeatability across preictal event windows. Top-attributed-channel masking is compared with random-channel masking of the same size; these are stability and perturbation-faithfulness checks, not clinical validation.

| SOP (min) | Patients | Events | Channel rank Spearman | Top-channel Jaccard | Top-mask risk drop | Random-mask risk drop | Top-mask advantage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 16 | 93 | 0.9 ± 0.1 | 0.8 ± 0.2 | 0.2 ± 0.3 | 0.0 ± 0.1 | 0.2 ± 0.2 |
| 10 | 16 | 93 | 0.9 ± 0.1 | 0.7 ± 0.3 | 0.2 ± 0.3 | 0.0 ± 0.1 | 0.1 ± 0.2 |
| 15 | 16 | 93 | 0.9 ± 0.1 | 0.7 ± 0.2 | 0.2 ± 0.3 | 0.0 ± 0.1 | 0.1 ± 0.2 |
| 30 | 16 | 93 | 0.9 ± 0.1 | 0.7 ± 0.2 | 0.2 ± 0.3 | 0.0 ± 0.1 | 0.2 ± 0.2 |

## Uncertainty and paired comparisons

Person-cluster bootstrap confidence intervals and paired participant-level model contrasts are generated from the completed unique-person table. The resampling unit is the held-out person (with chb01/chb21 combined), rather than windows or seizures. Direction counts show how many of the paired people favor each model for event sensitivity and FAR; they are descriptive. With only 15 people, intervals are exploratory and do not establish equivalence or correct for multiple model/horizon/operating-point comparisons.

See [`statistical_uncertainty_report.md`](statistical_uncertainty_report.md), [`person_bootstrap_confidence_intervals.csv`](person_bootstrap_confidence_intervals.csv), and [`paired_model_comparisons.csv`](paired_model_comparisons.csv).

## Interpretation and limits

A validation FAR budget does not cap the rate on unseen people. Use per-person observed FAR and the people-meeting-target count to assess alarm control; where rates exceed the target, this run shows a validation-to-test calibration gap and the sensitivity/FAR trade-off is not yet solved.

This is retrospective within-database evaluation on a limited pediatric cohort, not external validation or evidence of clinical utility. The multi-horizon, domain-fusion, subject-adversarial, CNN, LSTM, and CNN-LSTM components are established techniques; the defensible contribution is the controlled comparison and event-level operating-point protocol with explanation audits, not a claim that any single component is unprecedented. The 15 distinct people provide limited statistical power, so report case-wise results alongside aggregate summaries.

For unique-person mean ± SD, see [`multihorizon_loso_person_summary.csv`](multihorizon_loso_person_summary.csv); individual person-level values are in [`multihorizon_loso_per_person.csv`](multihorizon_loso_per_person.csv). Detailed held-out case rows remain in [`multihorizon_loso_per_patient.csv`](multihorizon_loso_per_patient.csv). For label counts and exclusions, see [`label_audit.csv`](label_audit.csv).
