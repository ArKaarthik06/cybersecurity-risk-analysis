"""
cert_evaluation.py  (v3)
========================
Evaluation that is meaningful for a 0.3 % positive-rate problem.

* "accuracy" alone is meaningless here: always predicting "normal" is 99.7 % accurate
  on true attack-day labels (95.3 % on the original whole-user labels).  We report
  it for completeness but headline PR-AUC, F1, balanced accuracy, recall at a fixed
  false-alert budget, and USER-LEVEL detection.
* Thresholds are never tuned on the test fold.  Operating points are either fixed
  (decision-theoretic, from costs) or chosen on training-fold predictions.
* Everything is computed out-of-fold with user-grouped, scenario-stratified folds.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix, brier_score_loss, log_loss)


def day_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"thr": thr, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": prec, "recall": tpr, "F1": 2 * prec * tpr / max(prec + tpr, 1e-12),
            "accuracy": (tp + tn) / len(y), "balanced_acc": (tpr + tnr) / 2,
            "FPR": fp / max(fp + tn, 1)}


def curve_metrics(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.argmax(f1[:-1]))
    return {"ROC_AUC": roc_auc_score(y, p), "PR_AUC": average_precision_score(y, p),
            "oracle_best_F1": float(f1[i]), "oracle_thr": float(th[i]),
            "brier": brier_score_loss(y, np.clip(p, 0, 1)), "ECE": ece(y, p)}


def ece(y, p, bins=15):
    """Expected calibration error with equal-mass bins (equal-width is useless at 0.3 % prevalence)."""
    order = np.argsort(p)
    ys, ps = y[order], p[order]
    e = 0.0
    for chunk in np.array_split(np.arange(len(p)), bins):
        e += len(chunk) / len(p) * abs(ys[chunk].mean() - ps[chunk].mean())
    return float(e)


def recall_at_fpr(y, p, fpr_budget):
    neg = np.sort(p[y == 0])
    thr = neg[int(np.ceil((1 - fpr_budget) * len(neg))) - 1] if len(neg) else 1.0
    return float((p[y == 1] > thr).mean()), float(thr)


def user_level(df, p, k_days=2, thr=0.5, window=30):
    """
    A user is ALERTED if >= k_days days inside any `window`-day span have p >= thr.
    Returns detection stats against the true insider set.
    """
    d = pd.DataFrame({"user": df["user"].values, "day": pd.to_datetime(df["day"]).values,
                      "p": p, "ins": (df["scenario"].values > 0)})
    d["hit"] = (d["p"] >= thr).astype(int)
    alerted = {}
    for u, g in d.groupby("user"):
        g = g.sort_values("day")
        s = g.set_index("day")["hit"].rolling(f"{window}D").sum()
        alerted[u] = bool((s >= k_days).any())
    truth = d.groupby("user")["ins"].max()
    a = pd.Series(alerted).reindex(truth.index)
    tp = int((a & truth).sum()); fn = int((~a & truth).sum()); fp = int((a & ~truth).sum()); tn = int((~a & ~truth).sum())
    n = len(truth)
    return {"users": n, "insiders": int(truth.sum()), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "accuracy": (tp + tn) / n, "balanced_acc": (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2}


def user_max_auc(df, p):
    d = pd.DataFrame({"user": df["user"].values, "p": p, "ins": (df["scenario"].values > 0)})
    u = d.groupby("user").agg(mx=("p", "max"), ins=("ins", "max"))
    return float(roc_auc_score(u["ins"], u["mx"]))
