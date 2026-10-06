import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, average_precision_score,
                             precision_recall_curve, auc)


def patient_metrics(y_true, scores, threshold=.5):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else float("nan")
    return {"sensitivity": sensitivity, "specificity": specificity, "f1": f1, "pr_auc": pr_auc}


def summarize(rows):
    import pandas as pd
    frame = pd.DataFrame(rows)
    metrics = ["sensitivity", "specificity", "f1", "pr_auc", "three_state_accuracy", "three_state_macro_f1"]
    summary = {f"{m}_mean": float(frame[m].mean()) for m in metrics}
    summary.update({f"{m}_sd": float(frame[m].std(ddof=1)) if len(frame) > 1 else 0.0 for m in metrics})
    return summary
