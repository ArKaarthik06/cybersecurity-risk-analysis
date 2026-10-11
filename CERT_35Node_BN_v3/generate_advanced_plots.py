"""
generate_advanced_plots.py
==========================
Generates a comprehensive 6-panel academic dashboard and supplementary 
publication figures for the End-Semester Project report:

Panel 1: Precision-Recall curves & operating points
Panel 2: ROC curves with zoomed operational low-FPR region (FPR < 1%)
Panel 3: Calibration curves with confidence intervals & ECE
Panel 4: Operational Alert Volume vs Attacks Detected (Analyst fatigue curve)
Panel 5: Threshold vs F1-score & False Alarm trade-off across all models
Panel 6: Per-Scenario Detection Breakdown (CERT Insider Scenarios 1, 2, 3, 4)
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, confusion_matrix

import cert_preprocessing as P
import cert_evaluation as EV
import v3_common as C

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def build_advanced_dashboard():
    print("Loading dataset and saved model predictions...")
    df = C.load_base()
    y = df.y.values
    scenarios = df["scenario"].values

    preds_path = os.path.join(C.RESULTS, "oof_all_models.npz")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing {preds_path}. Run compare_models.py first.")
    
    npz = np.load(preds_path)
    models = {
        "35-Node Bayesian Net": npz["35-Node_Bayesian_Net"],
        "Naive Bayes (Bernoulli)": npz["Naive_Bayes_(Bernoulli)"],
        "Random Forest (Balanced)": npz["Random_Forest_(Balanced)"],
        "Logistic Regression": npz["Logistic_Regression"],
    }

    colors = {
        "35-Node Bayesian Net": "#1f77b4",        # Royal Blue
        "Naive Bayes (Bernoulli)": "#ff7f0e",     # Amber Orange
        "Random Forest (Balanced)": "#2ca02c",    # Emerald Green
        "Logistic Regression": "#d62728",         # Crimson Red
    }
    linestyles = {
        "35-Node Bayesian Net": "-",
        "Naive Bayes (Bernoulli)": "--",
        "Random Forest (Balanced)": "-.",
        "Logistic Regression": ":",
    }
    lw_map = {
        "35-Node Bayesian Net": 2.4,
        "Naive Bayes (Bernoulli)": 1.8,
        "Random Forest (Balanced)": 1.8,
        "Logistic Regression": 1.8,
    }

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Probabilistic Reasoning vs Machine Learning Baselines\nCMU SEI CERT Insider Threat Dataset (330,285 User-Days, 5-Fold Stratified CV)",
                 fontsize=15, fontweight="bold", y=0.98)

    # -------------------------------------------------------------
    # Panel 1: Precision-Recall Curve (Highlighting Class Imbalance)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    for name, p in models.items():
        pr, rc, th = precision_recall_curve(y, p)
        f1 = 2 * pr * rc / (pr + rc + 1e-12)
        best_f1 = np.max(f1[:-1])
        ax1.plot(rc, pr, label=f"{name} (F1={best_f1:.3f})", color=colors[name],
                 ls=linestyles[name], lw=lw_map[name])
    ax1.axhline(y.mean(), color="grey", ls=":", alpha=0.7, label=f"Random Chance (0.29%)")
    ax1.set_title("A. Precision-Recall Curves (Class Prevalence: 0.29%)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Recall (True Threat Days Detected)", fontsize=10)
    ax1.set_ylabel("Precision (True Alerts / Total Alerts)", fontsize=10)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="lower left", fontsize=8.5, frameon=True)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------
    # Panel 2: ROC Curve Zoomed to Low-FPR Operational Region (<1%)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    for name, p in models.items():
        fpr, tpr, _ = roc_curve(y, p)
        # Compute Recall @ FPR = 0.5%
        rec05, _ = EV.recall_at_fpr(y, p, 0.005)
        ax2.plot(fpr * 100, tpr * 100, label=f"{name} (Rec@0.5%={rec05*100:.1f}%)",
                 color=colors[name], ls=linestyles[name], lw=lw_map[name])
    ax2.set_xlim(-0.02, 1.0)
    ax2.set_ylim(0, 102)
    ax2.set_title("B. High-Stakes Operational ROC Zoom (FPR < 1.0%)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("False Positive Rate (%) [Alert Budget]", fontsize=10)
    ax2.set_ylabel("Recall / True Positive Rate (%)", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------
    # Panel 3: Calibration Curves & ECE (Reliability Diagram)
    # -------------------------------------------------------------
    ax3 = axes[0, 2]
    for name, p in models.items():
        q = pd.qcut(p, 15, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q).mean()
        ece_score = EV.ece(y, p)
        ax3.plot(g.p, g.y, "o-", label=f"{name} (ECE={ece_score:.4f})",
                 color=colors[name], lw=lw_map[name], ms=4.5)
    max_p = 0.5
    ax3.plot([0, max_p], [0, max_p], "k:", alpha=0.6, label="Ideal Calibration")
    ax3.set_title("C. Probabilistic Calibration (Equal-Mass Quantiles)", fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Predicted Probability P(High Threat)", fontsize=10)
    ax3.set_ylabel("Empirical Attack Rate", fontsize=10)
    ax3.set_xlim(-0.01, 0.45)
    ax3.set_ylim(-0.01, 0.45)
    ax3.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------
    # Panel 4: Daily Analyst Alert Fatigue Workload Curve
    # -------------------------------------------------------------
    ax4 = axes[1, 0]
    thresholds = np.linspace(0.01, 0.99, 100)
    for name, p in models.items():
        daily_alerts = []
        recalls = []
        for th in thresholds:
            y_hat = (p >= th)
            n_alerts = np.sum(y_hat)
            rec = np.sum(y_hat & (y == 1)) / np.sum(y == 1)
            daily_alerts.append(n_alerts)
            recalls.append(rec * 100)
        ax4.plot(recalls, daily_alerts, label=name, color=colors[name],
                 ls=linestyles[name], lw=lw_map[name])
    ax4.set_yscale("log")
    ax4.set_title("D. Analyst Workload Curve (Total Alerts vs Recall Caught)", fontsize=11, fontweight="bold", pad=8)
    ax4.set_xlabel("Attacks Caught (% Recall)", fontsize=10)
    ax4.set_ylabel("Total System Alerts Generated (Log Scale)", fontsize=10)
    ax4.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax4.grid(True, alpha=0.3)

    # -------------------------------------------------------------
    # Panel 5: Threshold Sweep vs F1 Score
    # -------------------------------------------------------------
    ax5 = axes[1, 1]
    th_range = np.linspace(0.01, 0.95, 80)
    for name, p in models.items():
        f1_list = []
        for th in th_range:
            y_pred = (p >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            f1_list.append(f1)
        ax5.plot(th_range, f1_list, label=name, color=colors[name],
                 ls=linestyles[name], lw=lw_map[name])
    ax5.set_title("E. F1-Score Stability Across Decision Thresholds", fontsize=11, fontweight="bold", pad=8)
    ax5.set_xlabel("Decision Operating Threshold", fontsize=10)
    ax5.set_ylabel("F1 Score", fontsize=10)
    ax5.set_ylim(0, 1.0)
    ax5.legend(loc="lower center", fontsize=8.5, frameon=True)
    ax5.grid(True, alpha=0.3)

    # -------------------------------------------------------------
    # Panel 6: Per-Scenario Attack Detection Rate (%)
    # -------------------------------------------------------------
    ax6 = axes[1, 2]
    # Scenarios in CERT: 1 (Exfil), 2 (Sabotage), 3 (IP Theft / Removable), 4 (Privilege Escalation)
    scenario_names = {
        1: "Scen 1: Data Exfil\n(USB / Email)",
        2: "Scen 2: IT Sabotage\n(Admin/Ports)",
        3: "Scen 3: IP Theft\n(Departing)",
        4: "Scen 4: Unauthorized\n(Credential Abuse)"
    }
    scen_ids = [1, 2, 3, 4]
    x_scen = np.arange(len(scen_ids))
    bar_width = 0.20
    operating_threshold = 0.3

    for idx, (name, p) in enumerate(models.items()):
        det_rates = []
        for s in scen_ids:
            mask = (scenarios == s)
            if np.sum(mask) > 0:
                caught = np.sum((p[mask] >= operating_threshold))
                rate = 100.0 * caught / np.sum(mask)
            else:
                rate = 0.0
            det_rates.append(rate)
        ax6.bar(x_scen + idx * bar_width, det_rates, bar_width, label=name, color=colors[name], alpha=0.88)

    ax6.set_xticks(x_scen + bar_width * 1.5)
    ax6.set_xticklabels([scenario_names[s] for s in scen_ids], fontsize=8.5)
    ax6.set_title("F. CERT Threat Scenario Detection Breakdown (at Thr=0.3)", fontsize=11, fontweight="bold", pad=8)
    ax6.set_ylabel("Detection Rate (% of Attack Days Caught)", fontsize=10)
    ax6.set_ylim(0, 105)
    ax6.legend(loc="upper right", fontsize=8.0, frameon=True)
    ax6.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    dashboard_path = os.path.join(C.RESULTS, "advanced_comparison_dashboard.png")
    plt.savefig(dashboard_path, dpi=200)
    print(f"Saved 6-panel advanced dashboard to: {dashboard_path}")

    # Also build a second focused graphic: User-level false alarm distribution
    fig2, ax_user = plt.subplots(figsize=(10, 5))
    user_df = pd.DataFrame({
        "user": df["user"].values,
        "is_insider": (df["scenario"].values > 0).astype(int),
    })
    for name, p in models.items():
        user_df[name] = p

    u_agg = user_df.groupby("user").agg({
        "is_insider": "max",
        **{name: "max" for name in models}
    })
    
    # False alarms out of 930 normal users at varying k-day detection criteria
    fp_counts = []
    tp_counts = []
    for name in models:
        u_res = EV.user_level(df, models[name], k_days=2, thr=0.3)
        fp_counts.append(u_res["FP"])
        tp_counts.append(u_res["TP"])

    y_pos = np.arange(len(models))
    bars = ax_user.barh(y_pos, fp_counts, color=[colors[m] for m in models], height=0.55)
    ax_user.set_yticks(y_pos)
    ax_user.set_yticklabels(list(models.keys()), fontsize=11, fontweight="bold")
    ax_user.set_xlabel("Number of Innocent Users Falsely Flagged (Out of 930 Normal Employees)", fontsize=11)
    ax_user.set_title("Operational Alert Fatigue: False Positives at User Level (Threshold = 0.3)", fontsize=13, fontweight="bold")
    
    for bar in bars:
        w = bar.get_width()
        ax_user.text(w + 3, bar.get_y() + bar.get_height()/2, f"{int(w)} users",
                     va="center", ha="left", fontsize=10, fontweight="bold")
    ax_user.set_xlim(0, max(fp_counts) + 35)
    ax_user.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    user_plot_path = os.path.join(C.RESULTS, "user_level_false_alarms.png")
    plt.savefig(user_plot_path, dpi=180)
    print(f"Saved user-level false alarm chart to: {user_plot_path}")

if __name__ == "__main__":
    build_advanced_dashboard()
