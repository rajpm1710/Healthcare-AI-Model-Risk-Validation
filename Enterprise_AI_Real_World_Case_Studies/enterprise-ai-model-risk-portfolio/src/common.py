import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'

def metrics(y, p, threshold=0.5):
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'roc_auc': round(float(roc_auc_score(y, p)), 4), 'pr_auc': round(float(average_precision_score(y, p)), 4), 'brier_score': round(float(brier_score_loss(y, p)), 4), 'precision': round(float(tp / max(tp + fp, 1)), 4), 'recall': round(float(tp / max(tp + fn, 1)), 4), 'false_positive_rate': round(float(fp / max(fp + tn, 1)), 4), 'positive_rate': round(float(np.mean(pred)), 4)}

def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0] / len(reference)
    b = np.histogram(current, bins=edges)[0] / len(current)
    a, b = np.maximum(a, 1e-6), np.maximum(b, 1e-6)
    return round(float(np.sum((b - a) * np.log(b / a))), 4)

def write_report(name, report):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f'{name}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
