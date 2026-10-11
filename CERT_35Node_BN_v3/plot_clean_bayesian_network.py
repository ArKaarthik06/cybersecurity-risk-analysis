"""
plot_clean_bayesian_network.py
==============================
Generates a publication-quality, high-resolution vector and PNG diagram
of the 35-Node Directed Acyclic Graph (DAG) Bayesian Network for the CERT
Insider Threat project in LIGHT MODE with ZERO text or node overlapping.

Features:
- Left-to-Right clean flow across the 4 functional layers
- Clustered Layer 1 nodes aligned with their Layer 2 behavioral parents to minimize edge crossings
- Crisp light-mode palette with distinct layer badges
- Edge styling with alpha transparency and subtle curvature to avoid clutter
- Clear layer boundary cards and subtitles
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import cert_bn_builder as B
import v3_common as C

def generate_clean_dag():
    print("Generating clean 35-Node Bayesian Network DAG diagram in Light Mode...")

    # Node definitions & ordered grouping to minimize edge crossings
    # Layer 1: 21 Evidence nodes grouped by functional domain
    l1_groups = [
        # Auth group
        ("AfterHoursLogin", "Logon outside 09:00-17:00"),
        ("UnusualLoginFreq", "Anomalous logon count"),
        ("HighMachineCount", "Accessed >3 workstations"),
        # Time-based group
        ("WeekendLogin", "Unusual weekend logons"),
        ("DeviceAfterHours", "USB connected off-hours"),
        ("AfterHoursEmail", "Email sent off-hours"),
        ("AfterHoursWeb", "Web browsing off-hours"),
        # Media & File transfer group
        ("DeviceConnectActivity", "Unusual USB connections"),
        ("FileCopyToRemovable", "Files copied to USB"),
        ("HighFileCopyActivity", "Rapid mass file copies"),
        ("FileAccessCount", "Anomalous file reads"),
        ("HighDataMovement", "Large outbound email volume"),
        # Communication group
        ("ExternalEmail", "Emails to external domain"),
        ("UnusualEmailVolume", "Surge in sent email count"),
        ("ExternalLargeEmail", "External email size >1MB"),
        # Staging & Execution group
        ("ArchiveCreation", "Zip / 7z / rar creation"),
        ("ExeActivity", "Executable file activity"),
        ("UnusualFileTypes", "Uncommon file extensions"),
        # Flight risk & Shadow IT group
        ("JobSearchActivity", "Employment portal traffic"),
        ("CloudStorageAccess", "Dropbox/GoogleDrive upload"),
        ("UnusualWebActivity", "Anomalous HTTP volume"),
    ]
    l1_nodes = [item[0] for item in l1_groups]

    # Layer 2: 9 Behavioral Indicators (ordered vertically to match L1 clusters)
    l2_nodes = [
        "AuthAnomaly",
        "TimeBasedAnomaly",
        "RemovableMediaAnomaly",
        "DataAccessAnomaly",
        "DataMovementAnomaly",
        "CommunicationAnomaly",
        "StagingAnomaly",
        "FlightRiskAnomaly",
        "ShadowITAnomaly",
    ]

    # Layer 3: 4 Threat Hypotheses
    l3_nodes = [
        "UnauthorizedAccess",
        "DataExfiltration",
        "ITSabotage",
        "IPTheft",
    ]

    # Layer 4: Target
    l4_nodes = ["RiskLevel"]

    # Graph Dimensions
    # Figure size: 26 inches wide by 18 inches high (spacious, zero crowding)
    fig, ax = plt.subplots(figsize=(27, 18.5), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    # X coordinates for the 4 layers
    x_coords = {
        1: 0.14,
        2: 0.44,
        3: 0.74,
        4: 0.93,
    }

    # Y coordinates: evenly distributed with generous spacing
    # Available Y range: 0.08 to 0.90
    y_range = (0.07, 0.91)
    
    # Layer 1 positions (21 nodes)
    y_l1 = np.linspace(y_range[1], y_range[0], len(l1_nodes))
    pos = {n: (x_coords[1], y_l1[i]) for i, n in enumerate(l1_nodes)}

    # Layer 2 positions (9 nodes)
    y_l2 = np.linspace(y_range[1] - 0.03, y_range[0] + 0.03, len(l2_nodes))
    for i, n in enumerate(l2_nodes):
        pos[n] = (x_coords[2], y_l2[i])

    # Layer 3 positions (4 nodes)
    y_l3 = np.linspace(y_range[1] - 0.12, y_range[0] + 0.12, len(l3_nodes))
    for i, n in enumerate(l3_nodes):
        pos[n] = (x_coords[3], y_l3[i])

    # Layer 4 position (1 node)
    pos["RiskLevel"] = (x_coords[4], 0.50)

    # -------------------------------------------------------------
    # Draw Background Layer Boundary Cards
    # -------------------------------------------------------------
    card_boxes = [
        (0.02, 0.03, 0.24, 0.93, "LAYER 1: OBSERVABLE EVIDENCE\n(21 Binary Anomaly Indicators)", "#fdfcf9", "#e8e0d0"),
        (0.32, 0.03, 0.24, 0.93, "LAYER 2: BEHAVIORAL INDICATORS\n(9 Latent Intermediate Noisy-OR Nodes)", "#f8fbfe", "#d2e5f5"),
        (0.62, 0.03, 0.24, 0.93, "LAYER 3: THREAT HYPOTHESES\n(4 Causal Attack Vectors)", "#faf7fd", "#e5dcf4"),
        (0.88, 0.18, 0.11, 0.64, "LAYER 4: TARGET\n(Ternary Risk CPT)", "#fff8f8", "#fcd9d9"),
    ]

    for x_c, y_c, w_c, h_c, title_c, bg_c, border_c in card_boxes:
        rect = patches.FancyBboxPatch(
            (x_c, y_c), w_c, h_c, boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=bg_c, edgecolor=border_c, lw=1.5, zorder=1
        )
        ax.add_patch(rect)
        # Header text
        ax.text(x_c + w_c / 2, y_c + h_c - 0.025, title_c, fontsize=10.5, fontweight="bold",
                ha="center", va="top", color="#4a5568", zorder=2)

    # -------------------------------------------------------------
    # Draw Directed Edges with Subtle Curvature
    # -------------------------------------------------------------
    # Edge lists
    # Layer 1 -> Layer 2
    edges_l1_l2 = []
    for b in B.BEHAVIORAL_NODES:
        for p in B.BEHAVIORAL_PARENTS[b]:
            edges_l1_l2.append((p, b))

    # Layer 2 -> Layer 3
    edges_l2_l3 = []
    for t in B.THREAT_NODES:
        for p in B.THREAT_PARENTS[t]:
            edges_l2_l3.append((p, t))

    # Layer 3 -> Layer 4
    edges_l3_l4 = [(t, "RiskLevel") for t in B.THREAT_NODES]

    # Draw Layer 1 -> Layer 2 edges (Warm Amber / Slate Gray)
    for src, dst in edges_l1_l2:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        # Offset start and end to avoid touching text
        x_start = x1 + 0.075
        x_end = x2 - 0.085
        rad = 0.04 if y1 > y2 else (-0.04 if y1 < y2 else 0.0)
        ax.annotate(
            "", xy=(x_end, y2), xytext=(x_start, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#94a3b8", lw=1.2, alpha=0.60,
                connectionstyle=f"arc3,rad={rad}", mutation_scale=11
            ),
            zorder=3
        )

    # Draw Layer 2 -> Layer 3 edges (Blue / Purple)
    for src, dst in edges_l2_l3:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        x_start = x1 + 0.085
        x_end = x2 - 0.085
        rad = 0.03 if y1 > y2 else (-0.03 if y1 < y2 else 0.0)
        ax.annotate(
            "", xy=(x_end, y2), xytext=(x_start, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#6366f1", lw=1.6, alpha=0.75,
                connectionstyle=f"arc3,rad={rad}", mutation_scale=13
            ),
            zorder=4
        )

    # Draw Layer 3 -> Layer 4 edges (Crimson)
    for src, dst in edges_l3_l4:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        x_start = x1 + 0.085
        x_end = x2 - 0.045
        rad = 0.02 if y1 > y2 else (-0.02 if y1 < y2 else 0.0)
        ax.annotate(
            "", xy=(x_end, y2), xytext=(x_start, y1),
            arrowprops=dict(
                arrowstyle="-|>", color="#ef4444", lw=2.2, alpha=0.85,
                connectionstyle=f"arc3,rad={rad}", mutation_scale=15
            ),
            zorder=5
        )

    # -------------------------------------------------------------
    # Draw Nodes with Clean Modern Badges (Light Mode)
    # -------------------------------------------------------------
    # 1. Layer 1 Nodes (Observable Evidence)
    for name in l1_nodes:
        x, y = pos[name]
        # Pill box
        w_box, h_box = 0.145, 0.028
        rect = patches.FancyBboxPatch(
            (x - w_box/2, y - h_box/2), w_box, h_box,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            facecolor="#fffbf0", edgecolor="#d97706", lw=1.2, zorder=6
        )
        ax.add_patch(rect)
        ax.text(x, y, name, fontsize=8.6, fontweight="bold", ha="center", va="center",
                color="#78350f", zorder=7)

    # 2. Layer 2 Nodes (Behavioral Indicators)
    for name in l2_nodes:
        x, y = pos[name]
        w_box, h_box = 0.165, 0.048
        rect = patches.FancyBboxPatch(
            (x - w_box/2, y - h_box/2), w_box, h_box,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor="#f0f9ff", edgecolor="#0284c7", lw=1.5, zorder=6
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.008, name, fontsize=9.6, fontweight="bold", ha="center", va="center",
                color="#0369a1", zorder=7)
        ax.text(x, y - 0.012, "[Noisy-OR Latent]", fontsize=7.4, ha="center", va="center",
                color="#64748b", fontstyle="italic", zorder=7)

    # 3. Layer 3 Nodes (Threat Hypotheses)
    for name in l3_nodes:
        x, y = pos[name]
        w_box, h_box = 0.165, 0.055
        rect = patches.FancyBboxPatch(
            (x - w_box/2, y - h_box/2), w_box, h_box,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            facecolor="#fdf4ff", edgecolor="#9333ea", lw=1.8, zorder=6
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.009, name, fontsize=10.2, fontweight="bold", ha="center", va="center",
                color="#7e22ce", zorder=7)
        ax.text(x, y - 0.013, "[Attack Vector]", fontsize=7.6, ha="center", va="center",
                color="#6b21a8", fontstyle="italic", zorder=7)

    # 4. Layer 4 Node (RiskLevel Target)
    x, y = pos["RiskLevel"]
    w_box, h_box = 0.095, 0.085
    rect = patches.FancyBboxPatch(
        (x - w_box/2, y - h_box/2), w_box, h_box,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor="#dc2626", edgecolor="#991b1b", lw=2.5, zorder=6
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.020, "RiskLevel", fontsize=11.5, fontweight="bold", ha="center", va="center",
            color="#ffffff", zorder=7)
    ax.text(x, y - 0.004, "P(Low, Med, High)", fontsize=7.8, ha="center", va="center",
            color="#fee2e2", zorder=7)
    ax.text(x, y - 0.024, "[Learned CPT]", fontsize=7.4, ha="center", va="center",
            color="#fecaca", fontstyle="italic", zorder=7)

    # -------------------------------------------------------------
    # Title & Legend Block (Clean, Modern Typography)
    # -------------------------------------------------------------
    ax.text(0.50, 0.985, "35-Node Bayesian Network Architecture for Insider Threat Assessment",
            fontsize=16.5, fontweight="bold", ha="center", va="top", color="#0f172a")
    ax.text(0.50, 0.965, "Hierarchical Causal Topology with Noisy-OR Parameterization on CMU SEI CERT R4.2 (47 Directed Edges)",
            fontsize=11.5, ha="center", va="top", color="#475569")

    # Legend at Bottom
    legend_y = 0.015
    ax.text(0.14, legend_y, "● Observable Evidence: 21 Features", fontsize=9.5, fontweight="bold", color="#b45309", ha="center")
    ax.text(0.44, legend_y, "● Latent Behavioral Nodes: 9 Intermediate States", fontsize=9.5, fontweight="bold", color="#0284c7", ha="center")
    ax.text(0.74, legend_y, "● Threat Hypotheses: 4 Attack Classes", fontsize=9.5, fontweight="bold", color="#9333ea", ha="center")
    ax.text(0.93, legend_y, "● Target Risk Node", fontsize=9.5, fontweight="bold", color="#dc2626", ha="center")

    ax.set_xlim(0.00, 1.00)
    ax.set_ylim(0.00, 1.00)

    plt.tight_layout()
    
    # Save both high-res PNG and PDF
    out_png = os.path.join(C.RESULTS, "bayesian_network_35node_clean.png")
    out_pdf = os.path.join(C.RESULTS, "bayesian_network_35node_clean.pdf")
    plt.savefig(out_png, dpi=300, facecolor="#ffffff", bbox_inches="tight")
    plt.savefig(out_pdf, facecolor="#ffffff", bbox_inches="tight")
    print(f"Saved clean DAG image to: {out_png}")
    print(f"Saved vector PDF to: {out_pdf}")

if __name__ == "__main__":
    generate_clean_dag()
