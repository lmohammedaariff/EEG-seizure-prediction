from __future__ import annotations
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import load_dataset
from .model import MultiDomainNet
from .metrics import patient_metrics, summarize
from sklearn.metrics import accuracy_score, f1_score


STAGES = ["temporal_only", "temporal_spectral", "multi_domain", "multi_domain_attention", "complete"]
BASELINES = ["cnn", "lstm", "cnn_lstm"]


class _WindowDataset(torch.utils.data.Dataset):
    """Index source arrays lazily so a fold does not duplicate the full EEG tensors."""
    def __init__(self, arrays, indices, targets=()):
        self.arrays, self.indices, self.targets = arrays, np.asarray(indices), targets
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = int(self.indices[i])
        values = [torch.from_numpy(a[idx]) for a in self.arrays]
        return tuple(values + [torch.as_tensor(t[i]) for t in self.targets])


def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _cap(indices, y, cap, seed):
    rng = np.random.default_rng(seed)
    kept = []
    for label in np.unique(y[indices]):
        ix = indices[y[indices] == label]
        if cap and len(ix) > cap:
            ix = rng.choice(ix, cap, replace=False)
        kept.extend(ix.tolist())
    rng.shuffle(kept)
    return np.asarray(kept, dtype=int)


def run_loso(cfg: dict, horizon: int, stage: str = "complete", dataset=None) -> tuple[pd.DataFrame, dict]:
    if stage not in STAGES + BASELINES:
        raise ValueError(f"Unknown stage {stage}")
    _seed(cfg["training"]["seed"])
    raw, spectral, tf, meta = dataset if dataset is not None else load_dataset(cfg, horizon)
    valid = meta.label.to_numpy() >= 0
    # Prediction is preictal vs non-preictal; ictal examples are not used as negatives.
    predict_valid = valid & (meta.label.to_numpy() != 2)
    subjects = sorted(meta.loc[predict_valid, "patient_id"].unique())
    if len(subjects) < 2:
        raise RuntimeError("LOSO requires at least two eligible patients")
    device_cfg = cfg["training"]["device"]
    device = torch.device("cuda" if device_cfg == "auto" and torch.cuda.is_available()
                          else "cpu" if device_cfg == "auto" else device_cfg)
    outdir = Path(cfg["data"]["output"]) / f"horizon_{horizon}m" / stage
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold, held_out in enumerate(subjects):
        test_ix = np.flatnonzero(predict_valid & (meta.patient_id.to_numpy() == held_out))
        state_test_ix = np.flatnonzero(valid & (meta.patient_id.to_numpy() == held_out))
        train_ix = np.flatnonzero(valid & (meta.patient_id.to_numpy() != held_out))
        train_ix = _cap(train_ix, meta.label.to_numpy(), cfg["training"]["max_train_windows_per_class"],
                        cfg["training"]["seed"] + fold)
        test_ix = _cap(test_ix, meta.label.to_numpy(), cfg["training"]["max_test_windows"],
                       cfg["training"]["seed"] + 10000 + fold)
        state_test_ix = _cap(state_test_ix, meta.label.to_numpy(), cfg["training"]["max_test_windows"],
                             cfg["training"]["seed"] + 20000 + fold)
        if not len(train_ix) or not len(test_ix):
            continue
        train_subjects = sorted(meta.iloc[train_ix].patient_id.unique())
        subject_to_idx = {s: i for i, s in enumerate(train_subjects)}
        subject_y = np.array([subject_to_idx[s] for s in meta.iloc[train_ix].patient_id], dtype=np.int64)
        y_binary = (meta.label.to_numpy()[train_ix] == 1).astype(np.float32)
        y_binary_mask = (meta.label.to_numpy()[train_ix] != 2)
        y_state = meta.label.to_numpy()[train_ix].astype(np.int64)
        arrays = [raw, spectral, tf]
        targets = (y_binary, y_state, subject_y, y_binary_mask)
        ds = _WindowDataset(arrays, train_ix, targets)
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
                            num_workers=cfg["training"]["workers"], pin_memory=device.type == "cuda")
        domains = {"temporal_only": ("temporal",), "temporal_spectral": ("temporal", "spectral"),
                   "cnn": ("temporal",), "lstm": ("temporal",), "cnn_lstm": ("temporal",)}.get(
                       stage, ("temporal", "spectral", "time_frequency"))
        architecture = stage if stage in BASELINES else "cnn"
        fusion = "attention" if stage in ("multi_domain_attention", "complete") else "mean"
        model = MultiDomainNet(channels=raw.shape[1], width=cfg["model"]["width"],
                               dropout=cfg["model"]["dropout"], n_subjects=len(train_subjects),
                               fusion=fusion, active_domains=domains, temporal_architecture=architecture,
                               domain_adversarial=stage == "complete" and cfg["model"]["domain_adversarial"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                                      weight_decay=cfg["training"]["weight_decay"])
        state_loss = nn.CrossEntropyLoss()
        n_pos = max(1, int(((y_binary == 1) & y_binary_mask).sum()))
        n_neg = max(1, int(((y_binary == 0) & y_binary_mask).sum()))
        # Square-root weighting tempers class imbalance without letting a tiny positive class
        # dominate optimization; the 0.5 threshold remains fixed before held-out evaluation.
        pos_weight = torch.tensor(min(20.0, (n_neg / n_pos) ** 0.5), device=device)
        pred_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        subj_loss = nn.CrossEntropyLoss()
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            # Gradually increase adversarial pressure during training.
            strength = cfg["training"]["adversarial_weight"] * (epoch + 1) / cfg["training"]["epochs"]
            for batch in loader:
                a, b, c, yp, ys, yu, ym = [v.to(device, non_blocking=True) for v in batch]
                a, b, c = a.float(), b.float(), c.float()
                optimizer.zero_grad(set_to_none=True)
                result = model(a, b, c, strength)
                pred_logits = result["prediction_logits"][ym]
                loss = pred_loss(pred_logits, yp[ym]) + 0.3 * state_loss(result["state_logits"], ys)
                if result["subject_logits"] is not None:
                    loss = loss + subj_loss(result["subject_logits"], yu)
                loss.backward(); optimizer.step()
        model.eval()
        test_loader = DataLoader(_WindowDataset(arrays, test_ix), batch_size=cfg["training"]["batch_size"])
        scores = []
        with torch.no_grad():
            for a, b, c in test_loader:
                res = model(a.to(device).float(), b.to(device).float(), c.to(device).float())
                scores.extend(torch.sigmoid(res["prediction_logits"]).cpu().numpy().tolist())
        state_loader = DataLoader(_WindowDataset(arrays, state_test_ix), batch_size=cfg["training"]["batch_size"])
        state_predictions = []
        with torch.no_grad():
            for a, b, c in state_loader:
                res = model(a.to(device).float(), b.to(device).float(), c.to(device).float())
                state_predictions.extend(res["state_logits"].argmax(1).cpu().numpy().tolist())
        y_test = (meta.label.to_numpy()[test_ix] == 1).astype(int)
        metrics = patient_metrics(y_test, scores, cfg["training"]["threshold"])
        state_true = meta.label.to_numpy()[state_test_ix]
        metrics["three_state_accuracy"] = accuracy_score(state_true, state_predictions)
        metrics["three_state_macro_f1"] = f1_score(state_true, state_predictions, average="macro", labels=[0, 1, 2], zero_division=0)
        labels_all = meta.label.to_numpy()
        patients_all = meta.patient_id.to_numpy()
        rows.append({"patient_id": held_out, "horizon_minutes": horizon, "stage": stage,
                     "n_train_windows": len(train_ix), "n_test_windows": len(test_ix),
                     "n_train_excluded_windows": int(((labels_all < 0) & (patients_all != held_out)).sum()),
                     "n_test_excluded_windows": int(((labels_all < 0) & (patients_all == held_out)).sum()),
                     **metrics})
        pd.DataFrame({"patient_id": meta.patient_id.to_numpy()[test_ix],
                      "recording_id": meta.recording_id.to_numpy()[test_ix],
                      "window_id": meta.window_id.to_numpy()[test_ix],
                      "start_time": meta.start_time.to_numpy()[test_ix],
                      "end_time": meta.end_time.to_numpy()[test_ix],
                      "patient_start_time": meta.patient_start_time.to_numpy()[test_ix],
                      "patient_end_time": meta.patient_end_time.to_numpy()[test_ix],
                      "time_reference": meta.time_reference.to_numpy()[test_ix],
                      "seizure_id": meta.seizure_id.to_numpy()[test_ix],
                      "seizure_onset": meta.seizure_onset.to_numpy()[test_ix],
                      "seizure_end": meta.seizure_end.to_numpy()[test_ix],
                      "seizure_recording_id": meta.seizure_recording_id.to_numpy()[test_ix],
                      "seizure_onset_in_recording": meta.seizure_onset_in_recording.to_numpy()[test_ix],
                      "preictal_horizon": meta.preictal_horizon.to_numpy()[test_ix],
                      "label_reason": meta.label_reason.to_numpy()[test_ix], "true_label": y_test,
                      "score": scores, "predicted_label": np.array(scores) >= cfg["training"]["threshold"]}).to_csv(
                          outdir / f"{held_out}_predictions.csv", index=False)
        pd.DataFrame({"patient_id": meta.patient_id.to_numpy()[state_test_ix],
                      "recording_id": meta.recording_id.to_numpy()[state_test_ix],
                      "window_id": meta.window_id.to_numpy()[state_test_ix],
                      "start_time": meta.start_time.to_numpy()[state_test_ix],
                      "end_time": meta.end_time.to_numpy()[state_test_ix],
                      "patient_start_time": meta.patient_start_time.to_numpy()[state_test_ix],
                      "patient_end_time": meta.patient_end_time.to_numpy()[state_test_ix],
                      "time_reference": meta.time_reference.to_numpy()[state_test_ix],
                      "seizure_id": meta.seizure_id.to_numpy()[state_test_ix],
                      "seizure_onset": meta.seizure_onset.to_numpy()[state_test_ix],
                      "seizure_end": meta.seizure_end.to_numpy()[state_test_ix],
                      "preictal_horizon": meta.preictal_horizon.to_numpy()[state_test_ix],
                      "label_reason": meta.label_reason.to_numpy()[state_test_ix], "true_state": state_true,
                      "predicted_state": state_predictions}).to_csv(outdir / f"{held_out}_states.csv", index=False)
        torch.save({"model": model.cpu().state_dict(), "patient": held_out, "horizon": horizon,
                    "stage": stage, "config": cfg, "channels": raw.shape[1]}, outdir / f"{held_out}.pt")
        print(f"{horizon}m {stage} held-out {held_out}: F1={metrics['f1']:.3f}, PR-AUC={metrics['pr_auc']:.3f}")
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "patient_metrics.csv", index=False)
    summary = summarize(rows) if rows else {}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return frame, summary
