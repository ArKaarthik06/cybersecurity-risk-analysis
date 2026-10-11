"""
generate_eda_plots.py
=====================
Generates publication-quality Exploratory Data Analysis (EDA) figures
for the CERT dataset (per-user per-day activity logs).

Figures generated:
1. `eda_activity_distributions.png`:
   - Feature distribution comparison: Normal vs Malicious days (Logon, Device, File, Email, Web).
   - Shows heavy skew and distinct attack footprints.
2. `eda_temporal_and_scenarios.png`:
   - Attack timeline over the calendar year.
   - Attack day breakdown across CERT scenarios 1, 2, 3, 4.
   - Per-user activity volume distribution.
3. `eda_correlation_heatmap.png`:
   - Correlation heatmap among core raw activity features and insider threat targets.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import v3_common as C

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def build_eda():
    print("Loading user-day base dataset (330,285 rows)...")
    df = C.load_base()
    OUT = C.RESULTS + os.sep
    os.makedirs(OUT, exist_ok=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Attack days: {df.y.sum()} / {len(df)} ({100*df.y.mean():.2f}%)")

    # =========================================================================
    # Figure 1: 6-Panel Feature Activity Distributions (Normal vs Attack Days)
    # =========================================================================
    print("Generating Figure 1: Activity Distributions...")
    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9.5))
    fig1.suptitle("CERT Dataset EDA: Daily Activity Distributions (Normal vs Attack Days)\n[N = 330,285 User-Days across 1,000 Employees]",
                  fontsize=14, fontweight="bold", y=0.98)

    features_to_plot = [
        ("logon_count", "Daily Logon Count", 0, 15),
        ("after_hours_logon_count", "After-Hours Logon Count", 0, 8),
        ("device_connect_count", "Removable Media / USB Connects", 0, 12),
        ("file_copy_to_removable_count", "Files Copied to Removable Media", 0, 20),
        ("external_email_count", "External Email Sent Count", 0, 25),
        ("web_activity_count", "Web Browsing Activity Count", 0, 150),
    ]

    normal_df = df[df.y == 0]
    attack_df = df[df.y == 1]

    for idx, (col, title, xmin, xmax) in enumerate(features_to_plot):
        ax = axes1[idx // 3, idx % 3]
        
        # Plot KDE / histograms
        bins = np.linspace(xmin, xmax, 25)
        ax.hist(normal_df[col].clip(upper=xmax), bins=bins, density=True, alpha=0.55,
                color="#1f77b4", label=f"Normal Days (N={len(normal_df):,})")
        ax.hist(attack_df[col].clip(upper=xmax), bins=bins, density=True, alpha=0.75,
                color="#d62728", label=f"Attack Days (N={len(attack_df):,})")
        
        # Median / Mean annotations
        med_norm = normal_df[col].median()
        med_attk = attack_df[col].median()
        ax.axvline(med_norm, color="#1f77b4", linestyle="--", lw=1.5, alpha=0.9)
        ax.axvline(med_attk, color="#d62728", linestyle="--", lw=1.5, alpha=0.9)

        ax.set_title(f"{title}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Count per User-Day", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f1_path = os.path.join(OUT, "eda_activity_distributions.png")
    plt.savefig(f1_path, dpi=160)
    print(f"Saved: {f1_path}")

    # =========================================================================
    # Figure 2: Temporal Timeline, Scenario Breakdown & Class Imbalance
    # =========================================================================
    print("Generating Figure 2: Temporal Trends and Scenario Breakdown...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig2.suptitle("CERT Dataset EDA: Threat Dynamics, Calendar Timeline & Imbalance",
                  fontsize=14, fontweight="bold", y=0.99)

    # Subplot A: Class Imbalance Donut Chart
    ax_pie = axes2[0]
    counts = [len(normal_df), len(attack_df)]
    labels = [f"Normal Days\n({len(normal_df):,} | 99.71%)", f"Attack Days\n({len(attack_df):,} | 0.29%)"]
    colors_pie = ["#4a90e2", "#e74c3c"]
    wedges, texts, autotexts = ax_pie.pie(
        counts, labels=labels, colors=colors_pie, autopct="%1.2f%%",
        startangle=140, pctdistance=0.75, explode=(0, 0.15),
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    # Donut hole
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax_pie.add_artist(centre_circle)
    ax_pie.set_title("A. Extreme Class Imbalance\n(0.29% Attack Prevalence)", fontsize=11, fontweight="bold")

    # Subplot B: Scenario Breakdown
    ax_scen = axes2[1]
    # For attack days only (y == 1)
    attack_scens = df[df.y == 1]["scenario"].value_counts().sort_index()
    scen_map = {
        1: "Scenario 1\nData Exfiltration",
        2: "Scenario 2\nIT Sabotage",
        3: "Scenario 3\nIP Theft",
        4: "Scenario 4\nUnauth Access"
    }
    present_scens = list(attack_scens.index)
    scen_labels = [f"{scen_map.get(s, f'Scenario {s}')}\n(N={attack_scens[s]})" for s in present_scens]
    palette = ["#e67e22", "#9b59b6", "#2ecc71", "#e74c3c"][:len(present_scens)]
    bars = ax_scen.bar(np.arange(len(present_scens)), attack_scens.values, color=palette, width=0.55, edgecolor="#555", lw=0.8)
    ax_scen.set_xticks(np.arange(len(present_scens)))
    ax_scen.set_xticklabels(scen_labels, fontsize=9.5)
    ax_scen.set_title("B. Attack Days per Threat Scenario (y=1)", fontsize=11, fontweight="bold")
    ax_scen.set_ylabel("Malicious User-Days", fontsize=10)
    for bar in bars:
        h = bar.get_height()
        ax_scen.text(bar.get_x() + bar.get_width()/2, h + 8, f"{int(h)}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_scen.set_ylim(0, max(attack_scens.values) * 1.18)
    ax_scen.grid(True, axis="y", alpha=0.3)

    # Subplot C: Timeline of Insider Attacks Over Time
    ax_time = axes2[2]
    df["day_dt"] = pd.to_datetime(df["day"])
    monthly_attacks = df[df.y == 1].groupby(pd.Grouper(key="day_dt", freq="ME")).size()
    ax_time.plot(monthly_attacks.index, monthly_attacks.values, "o-", color="#c0392b", lw=2, ms=6)
    ax_time.fill_between(monthly_attacks.index, monthly_attacks.values, color="#e74c3c", alpha=0.25)
    ax_time.set_title("C. Monthly Attack Day Frequency Timeline", fontsize=11, fontweight="bold")
    ax_time.set_xlabel("Timeline (Calendar Months)", fontsize=10)
    ax_time.set_ylabel("Malicious Days Detected", fontsize=10)
    ax_time.grid(True, alpha=0.3)
    plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f2_path = os.path.join(OUT, "eda_temporal_and_scenarios.png")
    plt.savefig(f2_path, dpi=160)
    print(f"Saved: {f2_path}")

    # =========================================================================
    # Figure 3: Correlation Matrix Heatmap
    # =========================================================================
    print("Generating Figure 3: Correlation Heatmap...")
    fig3, ax3 = plt.subplots(figsize=(11, 9))
    core_cols = [
        "logon_count", "after_hours_logon_count", "weekend_logon_count",
        "device_connect_count", "file_activity_count", "file_copy_to_removable_count",
        "email_count", "external_email_count", "external_large_email_count",
        "web_activity_count", "cloud_storage_count", "job_search_count",
        "y"
    ]
    col_names = [
        "Logons", "After-Hrs Logons", "Weekend Logons",
        "USB Connects", "File Access", "Removable Copies",
        "Emails", "Ext Emails", "Large Ext Emails",
        "Web Activity", "Cloud Uploads", "Job Searches",
        "INSIDER ATTACK (y)"
    ]
    corr_mat = df[core_cols].corr()
    corr_mat.columns = col_names
    corr_mat.index = col_names

    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr_mat, mask=mask, cmap="coolwarm", vmin=-0.05, vmax=0.6,
                annot=True, fmt=".2f", annot_kws={"size": 8.5},
                square=True, linewidths=0.6, ax=ax3, cbar_kws={"shrink": 0.8})
    ax3.set_title("CERT Dataset EDA: Feature Cross-Correlation Matrix Heatmap\n(Showing Inter-Indicator Dependencies Modeling Justification)",
                  fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    f3_path = os.path.join(OUT, "eda_correlation_heatmap.png")
    plt.savefig(f3_path, dpi=180)
    print(f"Saved: {f3_path}")
    print("\nAll EDA plots successfully generated!")

if __name__ == "__main__":
    build_eda()
