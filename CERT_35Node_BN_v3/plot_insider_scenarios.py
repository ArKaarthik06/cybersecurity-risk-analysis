"""
plot_insider_scenarios.py
=========================
Generates visual case study figures for 3 malicious insiders corresponding 
to the interactive web dashboard panels:
  - Insider 1: JRG0207 (Scenario 1: Data Exfiltration via USB/Cloud)
  - Insider 2: RHL0992 (Scenario 2: Mass File Copy & IT Sabotage)
  - Insider 3: MSO0222 (Scenario 3: Departing Employee IP Theft / Job Search)

For each insider, generates a dedicated 3-panel figure:
  1. Risk Level Posteriors P(Low), P(Medium), P(High) bar
  2. Threat Hypotheses Probabilities % (Exfiltration, Unauthorized, Sabotage, IP Theft)
  3. Active Path DAG Subgraph: Triggered evidence nodes -> Latent behavioral anomalies -> Threat Hypotheses -> RiskLevel
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine
import cert_bn_builder as B
import v3_common as C

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

def build_insider_case_studies():
    print("Loading model parameters and samples...")
    param_path = os.path.join(C.DATA, "model_params.json")
    with open(param_path) as f:
        params = json.load(f)

    # Format params for factor building
    clean_params = {"nodes": {}}
    for n, spec in params["nodes"].items():
        spec = dict(spec)
        if "prior" in spec: spec["prior"] = {int(k): v for k, v in spec["prior"].items()}
        if "cpt" in spec: spec["cpt"] = {tuple(int(x) for x in k.split(",")): {int(a): b for a, b in v.items()} for k, v in spec["cpt"].items()}
        clean_params["nodes"][n] = spec

    bn = BayesianNetwork()
    factors = bn.build_factors(clean_params)
    ve = VariableEliminationEngine(bn, factors)

    # The 3 Insiders from samples.json
    samples_path = os.path.join(C.DATA, "samples.json")
    samples_data = json.load(open(samples_path))["samples"]
    insiders = [s for s in samples_data if s["is_insider"] == 1][:3]

    OUT = C.RESULTS + os.sep
    os.makedirs(OUT, exist_ok=True)

    for idx, ins in enumerate(insiders, 1):
        uid = ins["user"]
        day = ins["day"]
        scen_title = ins["label"]
        ev = {k: int(ins["evidence"].get(k, 0)) for k in B.EVIDENCE_NODES}
        active_ev = ins["active_evidence"]

        print(f"\nProcessing Insider {idx}: {uid} on {day}...")

        # 1. Compute Posteriors
        risk_dist, _ = ve.query(B.RISK_NODE, ev)
        p_low = risk_dist.get(0, 0.0)
        p_med = risk_dist.get(1, 0.0)
        p_high = risk_dist.get(2, 0.0)

        threat_posts = {}
        for t in B.THREAT_NODES:
            td, _ = ve.query(t, ev)
            threat_posts[t] = td.get(1, 0.0)

        beh_posts = {}
        for b in B.BEHAVIORAL_NODES:
            bd, _ = ve.query(b, ev)
            beh_posts[b] = bd.get(1, 0.0)

        # 2. Plotting 3-panel figure with spacious width and height to prevent text overlap
        fig = plt.figure(figsize=(26, 9.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.2, 3.2], wspace=0.25)

        fig.suptitle(f"Case Study: Malicious Insider {uid} ({day}) — {scen_title}",
                     fontsize=16, fontweight="bold", y=0.98)

        # -------------------------------------------------------------
        # Subplot 1: Risk Level Posterior Bar (Col 0)
        # -------------------------------------------------------------
        ax_risk = fig.add_subplot(gs[0, 0])
        risk_labels = ["Low Risk", "Medium Risk", "High Risk"]
        risk_vals = [p_low * 100, p_med * 100, p_high * 100]
        risk_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        bars_r = ax_risk.bar(risk_labels, risk_vals, color=risk_colors, width=0.48, edgecolor="#444", lw=0.8)
        ax_risk.set_title("A. Overall Risk Posterior P(Risk)", fontsize=12, fontweight="bold", pad=12)
        ax_risk.set_ylabel("Posterior Probability (%)", fontsize=11)
        ax_risk.set_ylim(0, 115)
        for b in bars_r:
            h = b.get_height()
            ax_risk.text(b.get_x() + b.get_width()/2, h + 2.5, f"{h:.1f}%",
                         ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax_risk.grid(True, axis="y", alpha=0.3)

        # -------------------------------------------------------------
        # Subplot 2: Threat Hypotheses Probabilities (Col 1)
        # -------------------------------------------------------------
        ax_threat = fig.add_subplot(gs[0, 1])
        t_names = ["DataExfiltration", "UnauthorizedAccess", "ITSabotage", "IPTheft"]
        t_display = ["Data Exfiltration", "Unauthorized Access", "IT Sabotage", "IP Theft"]
        t_vals = [threat_posts[t] * 100 for t in t_names]
        colors_t = ["#e74c3c" if v >= 50 else ("#f39c12" if v >= 20 else "#3498db") for v in t_vals]
        
        bars_t = ax_threat.barh(t_display, t_vals, color=colors_t, height=0.45, edgecolor="#444", lw=0.8)
        ax_threat.set_title("B. Threat Hypotheses P(Threat | Evidence)", fontsize=12, fontweight="bold", pad=12)
        ax_threat.set_xlabel("Probability (%)", fontsize=11)
        ax_threat.set_xlim(0, 118)
        ax_threat.axvline(50, color="red", ls=":", alpha=0.7, label="Critical Threshold (50%)")
        for b in bars_t:
            w = b.get_width()
            ax_threat.text(w + 2.5, b.get_y() + b.get_height()/2, f"{w:.1f}%",
                           va="center", ha="left", fontsize=11, fontweight="bold")
        ax_threat.legend(loc="lower right", fontsize=9.5)
        ax_threat.grid(True, axis="x", alpha=0.3)

        # -------------------------------------------------------------
        # Subplot 3: Directed Graph Activation Flow (Col 2)
        # -------------------------------------------------------------
        ax_graph = fig.add_subplot(gs[0, 2])
        ax_graph.axis("off")
        ax_graph.set_title("C. Causal DAG Activation Trace (Evidence -> Anomaly -> Threat -> Risk)",
                           fontsize=12, fontweight="bold", pad=12)

        # Layer coordinates with ample separation:
        # Layer 1: Evidence (x = 0.08)
        # Layer 2: Behavioral (x = 0.40)
        # Layer 3: Threat (x = 0.73)
        # Layer 4: Risk (x = 0.94)
        y_ev = np.linspace(0.88, 0.10, len(active_ev))
        ev_pos = {n: (0.08, y_ev[i]) for i, n in enumerate(active_ev)}

        # Layer 2: Relevant Behavioral Nodes
        rel_beh = [b for b in B.BEHAVIORAL_NODES if any(p in active_ev for p in B.BEHAVIORAL_PARENTS[b])]
        if not rel_beh: rel_beh = ["AuthAnomaly", "DataAccessAnomaly"]
        y_beh = np.linspace(0.86, 0.12, len(rel_beh))
        beh_pos = {b: (0.40, y_beh[i]) for i, b in enumerate(rel_beh)}

        # Layer 3: Top Threat Hypotheses
        top_threats = sorted(B.THREAT_NODES, key=lambda t: threat_posts[t], reverse=True)[:3]
        y_thr = np.linspace(0.78, 0.22, len(top_threats))
        thr_pos = {t: (0.73, y_thr[i]) for i, t in enumerate(top_threats)}

        # Layer 4: Risk Node
        risk_pos = {B.RISK_NODE: (0.94, 0.50)}

        # Draw Edges: Evidence -> Behavioral
        for e in active_ev:
            for b in rel_beh:
                if e in B.BEHAVIORAL_PARENTS[b]:
                    ax_graph.annotate("", xy=beh_pos[b], xytext=ev_pos[e],
                                      arrowprops=dict(arrowstyle="-|>", color="#ff7f0e", lw=1.6, alpha=0.85,
                                                      shrinkA=14, shrinkB=16))

        # Draw Edges: Behavioral -> Threats
        for b in rel_beh:
            for t in top_threats:
                if b in B.THREAT_PARENTS[t]:
                    ax_graph.annotate("", xy=thr_pos[t], xytext=beh_pos[b],
                                      arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=2.0, alpha=0.85,
                                                      shrinkA=16, shrinkB=18))

        # Draw Edges: Threats -> RiskLevel
        for t in top_threats:
            ax_graph.annotate("", xy=risk_pos[B.RISK_NODE], xytext=thr_pos[t],
                              arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.4, alpha=0.9,
                                              shrinkA=18, shrinkB=22))

        # Draw Nodes: Layer 1 Evidence (Amber pill boxes)
        for e, (x, y_c) in ev_pos.items():
            ax_graph.text(x, y_c, e, fontsize=8.2, fontweight="bold", ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#f39c12", lw=1.2))

        # Draw Nodes: Layer 2 Behavioral (Teal boxes with posterior)
        for b, (x, y_c) in beh_pos.items():
            b_val = beh_posts.get(b, 0.0) * 100
            ax_graph.text(x, y_c, f"{b}\n({b_val:.0f}%)", fontsize=8.8, fontweight="bold", ha="center", va="center",
                          bbox=dict(boxstyle="square,pad=0.45", facecolor="#d1ecf1", edgecolor="#17a2b8", lw=1.3))

        # Draw Nodes: Layer 3 Threat (Rose boxes with posterior)
        for t, (x, y_c) in thr_pos.items():
            t_val = threat_posts.get(t, 0.0) * 100
            ax_graph.text(x, y_c, f"{t}\n({t_val:.1f}%)", fontsize=9.2, fontweight="bold", ha="center", va="center",
                          bbox=dict(boxstyle="square,pad=0.5", facecolor="#f8d7da", edgecolor="#dc3545", lw=1.5))

        # Draw Node: Layer 4 Risk (Deep Red badge)
        ax_graph.text(risk_pos[B.RISK_NODE][0], risk_pos[B.RISK_NODE][1],
                      f"RiskLevel\nHigh ({p_high*100:.1f}%)", fontsize=10.0, fontweight="bold", ha="center", va="center",
                      color="white",
                      bbox=dict(boxstyle="round,pad=0.6", facecolor="#c0392b", edgecolor="#721c24", lw=2.2, alpha=0.95))

        # Layer Headers
        ax_graph.text(0.08, 0.98, "Observed Evidence", ha="center", fontsize=10, fontweight="bold", color="#444")
        ax_graph.text(0.40, 0.98, "Latent Anomaly", ha="center", fontsize=10, fontweight="bold", color="#444")
        ax_graph.text(0.73, 0.98, "Threat Hypothesis", ha="center", fontsize=10, fontweight="bold", color="#444")
        ax_graph.text(0.94, 0.98, "Overall Risk", ha="center", fontsize=10, fontweight="bold", color="#444")

        ax_graph.set_xlim(-0.04, 1.05)
        ax_graph.set_ylim(0.03, 1.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_case = os.path.join(OUT, f"insider_scenario_{idx}_{uid}.png")
        plt.savefig(out_case, dpi=180)
        print(f"Saved Insider {idx} plot to: {out_case}")

    print("\nAll 3 insider scenario case study figures successfully created!")

if __name__ == "__main__":
    build_insider_case_studies()
