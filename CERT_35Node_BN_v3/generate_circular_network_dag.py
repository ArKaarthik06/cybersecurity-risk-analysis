"""
generate_circular_network_dag.py
================================
Generates a circular-node Bayesian Network DAG matching the exact visual style
of the reference "26-Node Probabilistic Smart Home Decision Network DAG" image:
- Circular nodes with crisp black borders
- Sky blue for Evidence nodes (21)
- Soft green for Behavioral nodes (9)
- Orange for Threat Hypotheses (4)
- Purple for Risk Level (1)
- Strictly lined up LEVEL-WISE by causal hierarchy:
    * Layer 1: Evidence (21 nodes, arranged in two balanced columns)
    * Layer 2: Behavioral Indicators (9 nodes, lined up vertically)
    * Layer 3: Threat Hypotheses (4 nodes, lined up vertically)
    * Layer 4: Risk Level (1 node, centered at the terminal level)
- Minimalist gray connecting edges with subtle arrowheads
- Circular-marker Legend in the top-left corner
- Centered bold title at the top
- Guaranteed ZERO node or label overlapping
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import cert_bn_builder as B
import v3_common as C

def generate_network_plot():
    print("Constructing 35-node DAG with properly lined-up levels and circular nodes...")

    # 1. Build NetworkX DiGraph
    G = nx.DiGraph()

    evidence_nodes = list(B.EVIDENCE_NODES)
    behavioral_nodes = list(B.BEHAVIORAL_NODES)
    threat_nodes = list(B.THREAT_NODES)
    risk_nodes = [B.RISK_NODE]

    all_nodes = evidence_nodes + behavioral_nodes + threat_nodes + risk_nodes
    G.add_nodes_from(all_nodes)

    # Add edges
    for b in B.BEHAVIORAL_NODES:
        for p in B.BEHAVIORAL_PARENTS[b]:
            G.add_edge(p, b)

    for t in B.THREAT_NODES:
        for p in B.THREAT_PARENTS[t]:
            G.add_edge(p, t)

    for t in B.THREAT_NODES:
        G.add_edge(t, B.RISK_NODE)

    # Color mapping matching reference image
    color_map = {
        "Evidence": "#7cb5ec",      # Sky blue
        "Behavioral": "#70c173",    # Soft green
        "Threat": "#f59342",        # Warm orange
        "Risk": "#a38fc2",          # Soft purple
    }

    node_colors = {}
    for n in evidence_nodes: node_colors[n] = color_map["Evidence"]
    for n in behavioral_nodes: node_colors[n] = color_map["Behavioral"]
    for n in threat_nodes: node_colors[n] = color_map["Threat"]
    for n in risk_nodes: node_colors[n] = color_map["Risk"]

    # 2. Level-wise Structured Coordinate Assignment
    # Canvas coordinates:
    # Col 1: Outer Evidence (11 nodes) at X = -2.55
    # Col 2: Inner Evidence (10 nodes) at X = -1.20
    # Col 3: Behavioral Indicators (9 nodes) at X = 0.35
    # Col 4: Threat Hypotheses (4 nodes) at X = 1.75
    # Col 5: Risk Level (1 node) at X = 2.85
    pos = {}

    # Layer 4: Risk Level
    pos[B.RISK_NODE] = np.array([2.85, 0.0])

    # Layer 3: Threat Hypotheses (4 nodes, evenly spaced vertically)
    # Ordered to minimize edge crossings with Layer 2
    threat_order = [
        "UnauthorizedAccess",   # Connected to Auth, DataAccess, TimeBased
        "DataExfiltration",     # Connected to DataAccess, DataMovement, Comm, Removable, TimeBased
        "ITSabotage",           # Connected to Staging, DataAccess, Auth, Comm
        "IPTheft"               # Connected to FlightRisk, ShadowIT, DataMovement, Removable
    ]
    y_threats = np.linspace(1.50, -1.50, len(threat_order))
    for i, t in enumerate(threat_order):
        pos[t] = np.array([1.75, y_threats[i]])

    # Layer 2: Behavioral Indicators (9 nodes, lined up vertically)
    # Ordered top-to-bottom according to domain flow and parent affinity
    beh_order = [
        "AuthAnomaly",              # Top
        "TimeBasedAnomaly",
        "CommunicationAnomaly",
        "RemovableMediaAnomaly",
        "DataAccessAnomaly",        # Middle
        "DataMovementAnomaly",
        "StagingAnomaly",
        "ShadowITAnomaly",
        "FlightRiskAnomaly"         # Bottom
    ]
    y_beh = np.linspace(2.15, -2.15, len(beh_order))
    for i, b in enumerate(beh_order):
        pos[b] = np.array([0.35, y_beh[i]])

    # Layer 1: Evidence Nodes (21 nodes, in two clean columns)
    # Column 1 (Outer Evidence, 11 nodes)
    col1_nodes = [
        "AfterHoursLogin",      # Top (Auth)
        "UnusualLoginFreq",
        "HighMachineCount",
        "WeekendLogin",         # TimeBased
        "AfterHoursEmail",
        "AfterHoursWeb",
        "ExternalEmail",        # Comm & Flight
        "UnusualEmailVolume",
        "ArchiveCreation",      # Staging
        "ExeActivity",
        "UnusualFileTypes"      # Bottom (Staging)
    ]
    y_col1 = np.linspace(2.25, -2.25, len(col1_nodes))
    for i, n in enumerate(col1_nodes):
        pos[n] = np.array([-2.55, y_col1[i]])

    # Column 2 (Inner Evidence, 10 nodes)
    col2_nodes = [
        "DeviceAfterHours",      # Top (TimeBased & Removable)
        "DeviceConnectActivity", # Removable
        "FileCopyToRemovable",   # Removable & DataAccess & Movement
        "FileAccessCount",       # DataAccess
        "HighFileCopyActivity",  # DataAccess & Movement
        "HighDataMovement",      # Movement
        "CloudStorageAccess",    # ShadowIT
        "ExternalLargeEmail",    # ShadowIT
        "JobSearchActivity",     # FlightRisk
        "UnusualWebActivity"     # Bottom (ShadowIT & FlightRisk)
    ]
    y_col2 = np.linspace(2.05, -2.05, len(col2_nodes))
    for i, n in enumerate(col2_nodes):
        pos[n] = np.array([-1.20, y_col2[i]])

    # Figure creation with widescreen 16:9 proportion
    fig, ax = plt.subplots(figsize=(23, 14.5), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    # 3. Draw Edges (Subtle gray lines with crisp arrows, matching reference)
    for u, v in G.edges():
        p1 = pos[u]
        p2 = pos[v]
        ax.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(
                arrowstyle="-|>",
                color="#96a3ab",
                lw=1.15,
                alpha=0.68,
                shrinkA=25.5,   # Clean stop outside circle border
                shrinkB=25.5,   # Clean stop outside circle border
                mutation_scale=10.5
            ),
            zorder=1
        )

    # 4. Draw Circular Nodes
    node_radius = 0.170

    label_map = {
        "AfterHoursLogin": "AfterHours\nLogin",
        "UnusualLoginFreq": "UnusualLogin\nFreq",
        "HighMachineCount": "HighMachine\nCount",
        "WeekendLogin": "Weekend\nLogin",
        "DeviceAfterHours": "DeviceAfter\nHours",
        "AfterHoursEmail": "AfterHours\nEmail",
        "AfterHoursWeb": "AfterHours\nWeb",
        "ExternalEmail": "External\nEmail",
        "UnusualEmailVolume": "UnusualEmail\nVolume",
        "ExternalLargeEmail": "ExternalLarge\nEmail",
        "CloudStorageAccess": "CloudStorage\nAccess",
        "DeviceConnectActivity": "DeviceConnect\nActivity",
        "FileCopyToRemovable": "FileCopyTo\nRemovable",
        "HighFileCopyActivity": "HighFileCopy\nActivity",
        "FileAccessCount": "FileAccess\nCount",
        "HighDataMovement": "HighData\nMovement",
        "ArchiveCreation": "Archive\nCreation",
        "ExeActivity": "Exe\nActivity",
        "UnusualFileTypes": "UnusualFile\nTypes",
        "JobSearchActivity": "JobSearch\nActivity",
        "UnusualWebActivity": "UnusualWeb\nActivity",
        "AuthAnomaly": "Auth\nAnomaly",
        "TimeBasedAnomaly": "TimeBased\nAnomaly",
        "CommunicationAnomaly": "Communication\nAnomaly",
        "StagingAnomaly": "Staging\nAnomaly",
        "DataAccessAnomaly": "DataAccess\nAnomaly",
        "DataMovementAnomaly": "DataMovement\nAnomaly",
        "RemovableMediaAnomaly": "RemovableMedia\nAnomaly",
        "FlightRiskAnomaly": "FlightRisk\nAnomaly",
        "ShadowITAnomaly": "ShadowIT\nAnomaly",
        "UnauthorizedAccess": "Unauthorized\nAccess",
        "DataExfiltration": "Data\nExfiltration",
        "ITSabotage": "IT\nSabotage",
        "IPTheft": "IP\nTheft",
        "RiskLevel": "Risk\nLevel"
    }

    for n in G.nodes():
        x, y = pos[n]
        c = node_colors[n]

        # Draw circle with solid black border
        circle = patches.Circle(
            (x, y),
            radius=node_radius,
            facecolor=c,
            edgecolor="#000000",
            linewidth=1.75,
            zorder=3
        )
        ax.add_patch(circle)

        # Draw label centered inside circle
        lbl = label_map.get(n, n)
        fsize = 6.4 if "\n" in lbl else 7.2
        ax.text(
            x, y,
            lbl,
            fontsize=fsize,
            fontweight="bold",
            ha="center",
            va="center",
            color="#000000",
            zorder=4,
            linespacing=0.92
        )

    # 5. Title centered at the top (matching reference font and styling)
    plt.title(
        "35-Node Probabilistic Insider Threat Bayesian Network DAG",
        fontsize=16.5,
        fontweight="bold",
        pad=32,
        color="#000000"
    )

    # 6. Legend in top-left matching reference image EXACTLY with circular markers
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Evidence (21)",
               markerfacecolor=color_map["Evidence"], markeredgecolor="#000000", markeredgewidth=1.3, markersize=11.5),
        Line2D([0], [0], marker="o", color="w", label="Behavioral (9)",
               markerfacecolor=color_map["Behavioral"], markeredgecolor="#000000", markeredgewidth=1.3, markersize=11.5),
        Line2D([0], [0], marker="o", color="w", label="Threat Hypotheses (4)",
               markerfacecolor=color_map["Threat"], markeredgecolor="#000000", markeredgewidth=1.3, markersize=11.5),
        Line2D([0], [0], marker="o", color="w", label="Risk Level (1)",
               markerfacecolor=color_map["Risk"], markeredgecolor="#000000", markeredgewidth=1.3, markersize=11.5),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=11.5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        framealpha=0.95,
        borderpad=0.9,
        handletextpad=0.7,
        labelspacing=0.65
    )
    leg.get_frame().set_linewidth(0.8)

    # Set canvas limits with comfortable margins
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    # Leave extra room on the left so legend doesn't overlap the top nodes
    ax.set_xlim(-3.60, max(all_x) + 0.50)
    ax.set_ylim(min(all_y) - 0.45, max(all_y) + 0.45)

    plt.tight_layout()

    out_png = os.path.join(C.RESULTS, "bayesian_network_circular_dag.png")
    out_clean_png = os.path.join(C.RESULTS, "bayesian_network_35node_clean.png")
    out_pdf = os.path.join(C.RESULTS, "bayesian_network_circular_dag.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.savefig(out_clean_png, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="#ffffff")
    print(f"Saved level-wise circular DAG PNG: {out_png}")
    print(f"Saved clean DAG PNG for report: {out_clean_png}")
    print(f"Saved level-wise circular DAG PDF: {out_pdf}")

if __name__ == "__main__":
    generate_network_plot()
