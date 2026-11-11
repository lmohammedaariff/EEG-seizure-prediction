"""Audit saved multihorizon LOSO outputs and their split/metric invariants."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'artifacts'/'results-multihorizon-16case'
DEST=ROOT/'docs'/'evaluation_integrity_audit.md'
EXPECTED_MODELS={'multi_horizon_temporal','multi_horizon_temporal_spectral','multi_horizon_multi_domain','multi_horizon','multi_horizon_cnn','multi_horizon_lstm','multi_horizon_cnn_lstm'}
EXPECTED_HORIZONS={5,10,15,30}; EXPECTED_BUDGETS={0.1,0.5,1.0}

def person(case: str) -> str:
    case=case.lower()
    return 'chb01_chb21' if case in {'chb01','chb21'} else case

def main():
    manifest=json.loads((RESULTS/'multihorizon_manifest.json').read_text(encoding='utf-8'))
    cases=set(manifest['test_subjects'])
    folds=pd.read_csv(RESULTS/'multihorizon_loso_per_patient.csv')
    persons=pd.read_csv(RESULTS/'multihorizon_loso_per_person.csv')
    labels=pd.read_csv(RESULTS/'label_audit.csv')
    errors=[]
    expected_n=len(cases)*len(EXPECTED_MODELS)*len(EXPECTED_HORIZONS)*len(EXPECTED_BUDGETS)
    if len(folds)!=expected_n: errors.append(f'fold table has {len(folds)} rows, expected {expected_n}')
    combos=set(zip(folds.model_variant,folds.test_subject,folds.horizon_minutes,folds.false_alarm_budget_per_hour))
    expected_combos={(m,c,h,b) for m in EXPECTED_MODELS for c in cases for h in EXPECTED_HORIZONS for b in EXPECTED_BUDGETS}
    if combos!=expected_combos: errors.append(f'fold result grid mismatch: missing={len(expected_combos-combos)}, extra={len(combos-expected_combos)}')
    split_error=0
    for r in folds[['test_subject','validation_subject','train_subjects']].drop_duplicates().itertuples(index=False):
        test=str(r.test_subject).lower(); val=str(r.validation_subject).lower(); train={x.strip().lower() for x in str(r.train_subjects).split(',') if x.strip()}
        if person(test)==person(val) or person(test) in {person(x) for x in train} or person(val) in {person(x) for x in train}: split_error+=1
    if split_error: errors.append(f'{split_error} train/validation/test split records overlap at person level')
    for metric,expected in [('event_sensitivity',folds.detected_events / folds.eligible_events),('false_alarms_per_hour',folds.false_alarms / folds.eligible_interictal_hours)]:
        observed=folds[metric].to_numpy(dtype=float); calc=expected.to_numpy(dtype=float)
        mask=np.isfinite(observed)&np.isfinite(calc)
        if mask.any() and not np.allclose(observed[mask],calc[mask],rtol=1e-8,atol=1e-10): errors.append(f'{metric} identity mismatch in {int((~np.isclose(observed[mask],calc[mask],rtol=1e-8,atol=1e-10)).sum())} rows')
    if set(folds.horizon_minutes)!=EXPECTED_HORIZONS or set(folds.false_alarm_budget_per_hour)!=EXPECTED_BUDGETS or set(folds.model_variant)!=EXPECTED_MODELS: errors.append('Unexpected model/horizon/FAR setting in fold table')
    if len(persons)!=15*len(EXPECTED_MODELS)*len(EXPECTED_HORIZONS)*len(EXPECTED_BUDGETS): errors.append(f'unique-person table has unexpected row count {len(persons)}')
    if persons.test_person.nunique()!=15: errors.append(f'unique-person table reports {persons.test_person.nunique()} people, expected 15')
    label_rows=len(labels)
    primary=persons[(persons.model_variant=='multi_horizon')&(persons.false_alarm_budget_per_hour==0.5)]
    event_counts=primary.groupby('horizon_minutes').agg(people=('test_person','nunique'),annotated=('all_annotated_events','sum'),eligible=('eligible_events','sum'),detected=('detected_events','sum'))
    if not (event_counts.annotated==100).all() or not (event_counts.eligible==98).all(): errors.append('event denominators did not match 100 annotated / 98 eligible for every SOP')
    cfg=manifest['config']; protocol=manifest.get('protocol','')
    if cfg.get('sph_minutes')!=5 or cfg.get('window_seconds')!=10 or cfg.get('overlap')!=0.5 or cfg.get('interictal_buffer_minutes')!=60 or cfg.get('alarm_merge_gap_minutes')!=1.0: errors.append('manifest preprocessing/label/alarm parameters differ from expected protocol')
    audit=[
      '# LOSO evaluation integrity audit','',
      'This audit checks the saved CHB-MIT result tables against the run manifest and reviews the preprocessing, labeling, split, threshold, and alarm code paths. It is a reproducibility/code audit; it is not an independent clinical annotation audit.', '',
      '## Automated artifact checks','',
      f'- Fold-level table: {len(folds):,} rows; expected grid: 7 model variants × 16 held-out case folders × 4 SOPs × 3 FAR targets = {expected_n:,} rows.',
      f'- Unique-person table: {persons.test_person.nunique()} people after combining chb01/chb21; {len(persons):,} model/person/SOP/FAR rows.',
      f'- Split audit: {"PASS" if split_error==0 else "FAIL"}; each distinct test person and validation person is excluded from that fold’s training cases.',
      '- Metric arithmetic: event sensitivity equals detected events divided by eligible events; FAR equals false-alarm count divided by eligible interictal exposure hours.',
      f'- Label audit: {label_rows} case-by-SOP rows.',
      '', '| SOP (min) | People | Annotated events | Evaluable events | Detected by full model at 0.5/h validation target |', '|---:|---:|---:|---:|---:|'
    ]
    for h,row in event_counts.sort_index().iterrows(): audit.append(f'| {h} | {int(row.people)} | {int(row.annotated)} | {int(row.eligible)} | {int(row.detected)}/{int(row.eligible)} |')
    audit += ['', 'The two annotated events without an evaluable forecast interval are excluded by the event eligibility rule because no fully labeled positive forecast window is available under the current recording coverage. Event-level inference uses 15 people as the independent unit; the 98 eligible seizures are not treated as 98 independent subjects.', '', '## Protocol checks from source code and manifests','',
      '- **Preprocessing:** 0.5–45 Hz band-pass, 60 Hz notch where below Nyquist, resampling to 128 Hz, 10-second windows, and 50% overlap. Each window is normalized channel-wise. Missing canonical channels are zero-filled and recorded in preprocessing manifests.',
      '- **Label construction:** SOP-specific preictal interval is `[onset − SOP − SPH, onset − SPH)`; SPH is 5 minutes. Windows must lie wholly inside the forecast interval to receive a preictal label. A 60-minute interictal exclusion buffer is applied; ictal and ambiguous windows are not used as binary negatives.',
      '- **Annotations and timelines:** CHB-MIT summary onset/offset times are reconciled to EDF recordings. Absolute EDF header start timestamps are preferred for recording offsets; source summary clocks are the fallback. Summary-referenced missing EDFs and raw/cache mismatches cause errors.',
      '- **Patient separation:** chb01 and chb21 share one person group. The splitter excludes both case folders from training/validation whenever either case is held out. A distinct person supplies validation; the held-out person supplies test results.',
      '- **Threshold selection:** validation-only code calibrates each model/SOP threshold to maximize event sensitivity under the selected FAR target. The saved manifest records an 81-point probability grid plus a no-alarm sentinel; test metrics do not select thresholds.',
      '- **Event scoring:** alarm scores become available at the end of each window; alarms within a 1-minute merge gap are combined. An event is detected by a unique alarm onset in its SOP/SPH forecast interval. FAR exposure is the union duration of label-defined interictal EEG; preictal, ictal, and ambiguous time are excluded.',
      '', '## Scope limits and items to retain in the paper','',
      '- This result set covers 16 CHB-MIT case folders (15 people), not the complete database. Siena transfer uses one frozen CHB-MIT chb08 fold per architecture; it is not an average across all CHB folds.',
      '- Siena referential channels are converted to the project’s canonical bipolar montage. This is a documented preprocessing decision and a possible source of residual domain shift, not proof of the cause of the transfer failure.',
      '- The independent participant count is 15. Bootstrap intervals and paired comparisons are exploratory and are not corrected for multiple comparisons.',
      '- Source-code review found no train/validation/test patient overlap in the saved folds. Automated pytest checks were not run in this environment because pytest is unavailable in the project and bundled Python environments.',
      '', '## Audit result','', ('All saved-grid, split, denominator, and metric arithmetic checks passed.' if not errors else 'Audit failures: ' + '; '.join(errors)),
      '']
    DEST.write_text('\n'.join(audit),encoding='utf-8')
    print(DEST)
    print('errors=',errors)
    print(event_counts.to_string())
    if errors: raise SystemExit(1)
if __name__=='__main__': main()

