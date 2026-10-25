"""Measured-pixel binary classification metrics; pseudo-labels are never evaluated."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    confusion: tuple


def evaluate_measured(prediction, measured, *, ignore_index=2) -> BinaryMetrics:
    pred = np.asarray(prediction).reshape(-1)
    truth = np.asarray(measured).reshape(-1)
    if len(pred) != len(truth):
        raise ValueError("prediction and measured must have the same number of pixels")
    valid = (truth != ignore_index) & ((truth == 0) | (truth == 1))
    p, y = pred[valid].astype(int), truth[valid].astype(int)
    if not len(y):
        raise ValueError("No verified class-0/1 pixels available for evaluation")
    if not np.isin(p, [0,1]).all():
        raise ValueError("Predictions must contain only classes 0 and 1")
    # Confusion matrix ordered by true class 0/1 (rows) and prediction 0/1 (cols).
    cm = np.asarray([[( (y==i) & (p==j)).sum() for j in (0,1)] for i in (0,1)])
    precisions, recalls, f1s = [], [], []
    for i in (0,1):
        tp = cm[i,i]
        precision = tp / cm[:,i].sum() if cm[:,i].sum() else 0.
        recall = tp / cm[i,:].sum() if cm[i,:].sum() else 0.
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return BinaryMetrics(
        accuracy=float((p==y).mean()),
        macro_f1=float(np.mean(f1s)),
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        confusion=tuple(tuple(int(x) for x in row) for row in cm),
    )
