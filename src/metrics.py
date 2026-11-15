"""Metryki kliniczne: czułość i swoistość per klasa (one-vs-rest), F1, ROC AUC, bootstrap CI."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.data import CLASSES


def per_class_table(y_true, probs):
    """Dla każdej klasy: czułość, swoistość, precyzja, F1, AUC (klasa kontra reszta)."""
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    rows = []
    for k, cls in enumerate(CLASSES):
        tp = cm[k, k]
        fn = cm[k].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn)
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rows.append({
            'klasa': cls,
            'n': int(cm[k].sum()),
            'czulosc': sens,
            'swoistosc': tn / (tn + fp),
            'precyzja': prec,
            'f1': 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0,
            'auc': roc_auc_score(y_true == k, probs[:, k]),
        })
    return rows


def summary(y_true, probs):
    """Metryki zbiorcze używane do porównania modeli."""
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(1)
    table = per_class_table(y_true, probs)
    return {
        'accuracy': float((y_pred == y_true).mean()),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'macro_recall': float(np.mean([r['czulosc'] for r in table])),
        'macro_specificity': float(np.mean([r['swoistosc'] for r in table])),
        'macro_auc': float(roc_auc_score(y_true, probs, multi_class='ovr', average='macro')),
    }


def bootstrap_ci(y_true, probs, n_boot=1000, seed=0, alpha=0.05):
    """Przedziały ufności 95% metodą percentylową: losujemy test ze zwracaniem n_boot razy.

    Zwraca słownik metryka -> (dolna granica, górna granica) dla accuracy, macro-F1,
    macro AUC oraz czułości i swoistości każdej klasy."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    samples = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pr = y_true[idx], probs[idx]
        if len(np.unique(yt)) < len(CLASSES):
            continue  # AUC wymaga wszystkich klas w próbie
        values = summary(yt, pr)
        for row in per_class_table(yt, pr):
            values[f"czulosc_{row['klasa']}"] = row['czulosc']
            values[f"swoistosc_{row['klasa']}"] = row['swoistosc']
        for key, v in values.items():
            samples.setdefault(key, []).append(v)
    return {key: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
            for key, v in samples.items()}


def expected_calibration_error(y_true, probs, n_bins=10):
    """ECE: średnia (ważona liczbą próbek) różnica między pewnością a trafnością w przedziałach."""
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(y_true)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
