"""
compare_models.py
=================
Compare the 35-Node Bayesian Network against 3 machine learning algorithms:
  1. Bernoulli Naive Bayes (Probabilistic baseline)
  2. Random Forest (Balanced, Ensemble baseline)
  3. Logistic Regression (Balanced, Linear discriminative baseline)

All models are evaluated on the exact same 5-fold user-grouped split and 
identical 21 binary evidence features.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve

import cert_preprocessing as P
import cert_bn_builder as B
import cert_evaluation as EV
import v3_common as C

def run_comparison():
    print("=" * 75)
    print("  CERT INSIDER THREAT: 35-NODE BAYESIAN NETWORK vs ML BASELINES")
    print("=" * 75)

    df = C.load_base()
    y = df.y.values
    folds = list(P.grouped_folds(df))
    n_samples = len(df)
    n_pos = int(y.sum())
    print(f"Total instances: {n_samples:,} user-days | Attack instances: {n_pos:,} ({100*n_pos/n_samples:.2f}%)")
    print(f"Number of user-grouped folds: {len(folds)}")

    # 1. Load Bayesian Network out-of-fold predictions
    bn_oof_path = os.path.join(C.RESULTS, "oof_bn.npy")
    if not os.path.exists(bn_oof_path):
        raise FileNotFoundError(f"Missing {bn_oof_path}. Run cv_bn.py first.")
    oof_bn = np.load(bn_oof_path)
    print("Loaded existing out-of-fold predictions for 35-Node Bayesian Network.")

    # 2. Initialize ML models
    models = {
        "Naive Bayes (Bernoulli)": lambda: BernoulliNB(alpha=1.0),
        "Random Forest (Balanced)": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        ),
    }

    oof_preds = {name: np.zeros(n_samples) for name in models}

    # 3. 5-fold Cross-Validation
    print("\nTraining and evaluating ML baselines across 5 user-grouped folds...")
    t0 = time.time()
    for fi, (tr, te) in enumerate(folds):
        f_start = time.time()
        # Fit threshold flag rules on training fold to prevent leakage
        R = P.FlagRules().fit(df.iloc[tr])
        Ftr = R.transform(df.iloc[tr])
        Fte = R.transform(df.iloc[te])

        X_tr = Ftr[B.EVIDENCE_NODES].values.astype(float)
        y_tr = df.iloc[tr].y.values
        X_te = Fte[B.EVIDENCE_NODES].values.astype(float)

        for name, model_fn in models.items():
            clf = model_fn()
            clf.fit(X_tr, y_tr)
            # predict probability of attack (class 1)
            oof_preds[name][te] = clf.predict_proba(X_te)[:, 1]

        print(f"  Fold {fi+1}/5 finished in {time.time()-f_start:.1f}s")

    print(f"All ML baselines finished in {time.time()-t0:.1f}s\n")

    # Combine all predictions
    all_preds = {"35-Node Bayesian Net": oof_bn, **oof_preds}

    # Save out-of-fold predictions
    save_dict = {k.replace(" ", "_"): v for k, v in all_preds.items()}
    np.savez_compressed(os.path.join(C.RESULTS, "oof_all_models.npz"), **save_dict)

    # 4. Compute metrics
    summary_rows = []
    print("=" * 88)
    print(f"{'Model':<26} {'PR-AUC':>8} {'ROC-AUC':>8} {'Best F1':>8} {'Rec@0.1%':>9} {'Rec@0.5%':>9} {'ECE':>7} {'Brier':>7}")
    print("=" * 88)

    for name, p in all_preds.items():
        c_met = EV.curve_metrics(y, p)
        r01, _ = EV.recall_at_fpr(y, p, 0.001)
        r05, _ = EV.recall_at_fpr(y, p, 0.005)
        user_res = EV.user_level(df, p, k_days=2, thr=0.3)
        user_auc = EV.user_max_auc(df, p)

        row = {
            "Model": name,
            "PR_AUC": c_met["PR_AUC"],
            "ROC_AUC": c_met["ROC_AUC"],
            "Best_F1": c_met["oracle_best_F1"],
            "Recall@0.1%FPR": r01,
            "Recall@0.5%FPR": r05,
            "ECE": c_met["ECE"],
            "Brier": c_met["brier"],
            "User_AUC": user_auc,
            "User_TP": user_res["TP"],
            "User_FP": user_res["FP"],
        }
        summary_rows.append(row)

        print(f"{name:<26} {row['PR_AUC']:>8.4f} {row['ROC_AUC']:>8.4f} {row['Best_F1']:>8.4f} "
              f"{row['Recall@0.1%FPR']:>9.4f} {row['Recall@0.5%FPR']:>9.4f} {row['ECE']:>7.4f} {row['Brier']:>7.4f}")
    print("=" * 88)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(C.RESULTS, "model_comparison_metrics.csv"), index=False)

    # 5. Generate Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    colors = {
        "35-Node Bayesian Net": "#1f77b4",
        "Naive Bayes (Bernoulli)": "#ff7f0e",
        "Random Forest (Balanced)": "#2ca02c",
        "Logistic Regression": "#d62728",
    }
    linestyles = {
        "35-Node Bayesian Net": "-",
        "Naive Bayes (Bernoulli)": "--",
        "Random Forest (Balanced)": "-.",
        "Logistic Regression": ":",
    }

    # Plot 1: Precision-Recall Curve
    ax_pr = axes[0, 0]
    for name, p in all_preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ap = summary_df.loc[summary_df["Model"] == name, "PR_AUC"].values[0]
        ax_pr.plot(rc, pr, label=f"{name} (PR-AUC={ap:.3f})", color=colors[name], ls=linestyles[name], lw=2)
    ax_pr.set_title("Precision-Recall Curve (Day-Level, 0.3% Attack Rate)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="lower left", fontsize=9)

    # Plot 2: ROC Curve
    ax_roc = axes[0, 1]
    for name, p in all_preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = summary_df.loc[summary_df["Model"] == name, "ROC_AUC"].values[0]
        ax_roc.plot(fpr, tpr, label=f"{name} (ROC-AUC={auc:.3f})", color=colors[name], ls=linestyles[name], lw=2)
    ax_roc.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right", fontsize=9)

    # Plot 3: Calibration Plot (Reliability Diagram)
    ax_cal = axes[1, 0]
    for name, p in all_preds.items():
        q = pd.qcut(p, 15, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q).mean()
        ece_val = summary_df.loc[summary_df["Model"] == name, "ECE"].values[0]
        ax_cal.plot(g.p, g.y, "o-", label=f"{name} (ECE={ece_val:.4f})", color=colors[name], lw=1.5, ms=5)
    ax_cal.plot([0, 0.4], [0, 0.4], "k:", alpha=0.5, label="Perfect Calibration")
    ax_cal.set_title("Calibration Curve (Equal-Mass Quantiles)", fontsize=12, fontweight="bold")
    ax_cal.set_xlabel("Predicted Probability", fontsize=11)
    ax_cal.set_ylabel("Observed Attack Rate", fontsize=11)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.legend(loc="upper left", fontsize=9)

    # Plot 4: Summary Bar Chart of Key Metrics
    ax_bar = axes[1, 1]
    metrics_to_plot = ["PR_AUC", "Best_F1", "Recall@0.1%FPR"]
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    model_names = list(all_preds.keys())

    for idx, name in enumerate(model_names):
        vals = [summary_df.loc[summary_df["Model"] == name, m].values[0] for m in metrics_to_plot]
        ax_bar.bar(x + idx * width, vals, width, label=name, color=colors[name])

    ax_bar.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax_bar.set_xticklabels(["PR-AUC", "Best F1", "Recall @ 0.1% FPR"], fontsize=11)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Operational Detection Comparison", fontsize=12, fontweight="bold")
    ax_bar.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    plot_path = os.path.join(C.RESULTS, "model_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved comparison plot to: {plot_path}")
    print(f"Saved comparison metrics CSV to: {os.path.join(C.RESULTS, 'model_comparison_metrics.csv')}")

if __name__ == "__main__":
    run_comparison()
