"""
visualize_bn.py  (CERT Insider Threat Edition)
===============================================
Renders the 35-node CERT Insider Threat Bayesian Network as a
layered directed graph and saves it as a PNG.

Requires: matplotlib, networkx
"""

import os
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

# ---------------------------------------------------------------------------
# Network topology (mirrors cert_bn_builder.py exactly)
# ---------------------------------------------------------------------------

LAYER1_EVIDENCE = [
    "AfterHoursLogin",
    "UnusualLoginFreq",
    "WeekendLogin",
    "HighMachineCount",
    "DeviceConnectActivity",
    "DeviceAfterHours",
    "FileAccessCount",
    "FileCopyToRemovable",
    "ArchiveCreation",
    "ExeActivity",
    "UnusualFileTypes",
    "ExternalEmail",
    "UnusualEmailVolume",
    "AfterHoursEmail",
    "ExternalLargeEmail",
    "UnusualWebActivity",
    "AfterHoursWeb",
    "CloudStorageAccess",
    "JobSearchActivity",
    "HighDataMovement",
    "HighFileCopyActivity",
]

LAYER2_BEHAVIORAL = [
    "AuthAnomaly",
    "DataAccessAnomaly",
    "DataMovementAnomaly",
    "CommunicationAnomaly",
    "RemovableMediaAnomaly",
    "TimeBasedAnomaly",
    "StagingAnomaly",
    "FlightRiskAnomaly",
    "ShadowITAnomaly",
]

LAYER3_THREATS = [
    "DataExfiltration",
    "UnauthorizedAccess",
    "ITSabotage",
    "IPTheft",
]

LAYER4_RISK = ["RiskLevel"]

EDGES = [
    # Layer 1 → Layer 2
    ("AfterHoursLogin",       "AuthAnomaly"),
    ("UnusualLoginFreq",      "AuthAnomaly"),
    ("HighMachineCount",      "AuthAnomaly"),
    ("FileAccessCount",       "DataAccessAnomaly"),
    ("FileCopyToRemovable",   "DataAccessAnomaly"),
    ("HighFileCopyActivity",  "DataAccessAnomaly"),
    ("FileCopyToRemovable",   "DataMovementAnomaly"),
    ("HighDataMovement",      "DataMovementAnomaly"),
    ("HighFileCopyActivity",  "DataMovementAnomaly"),
    ("ExternalEmail",         "CommunicationAnomaly"),
    ("UnusualEmailVolume",    "CommunicationAnomaly"),
    ("DeviceConnectActivity", "RemovableMediaAnomaly"),
    ("FileCopyToRemovable",   "RemovableMediaAnomaly"),
    ("WeekendLogin",          "TimeBasedAnomaly"),
    ("AfterHoursEmail",       "TimeBasedAnomaly"),
    ("AfterHoursWeb",         "TimeBasedAnomaly"),
    ("DeviceAfterHours",      "TimeBasedAnomaly"),
    ("ArchiveCreation",       "StagingAnomaly"),
    ("ExeActivity",           "StagingAnomaly"),
    ("UnusualFileTypes",      "StagingAnomaly"),
    ("JobSearchActivity",     "FlightRiskAnomaly"),
    ("ExternalEmail",         "FlightRiskAnomaly"),
    ("CloudStorageAccess",    "ShadowITAnomaly"),
    ("ExternalLargeEmail",    "ShadowITAnomaly"),
    # Layer 2 → Layer 3
    ("DataAccessAnomaly",     "DataExfiltration"),
    ("DataMovementAnomaly",   "DataExfiltration"),
    ("CommunicationAnomaly",  "DataExfiltration"),
    ("RemovableMediaAnomaly", "DataExfiltration"),
    ("AuthAnomaly",           "UnauthorizedAccess"),
    ("DataAccessAnomaly",     "UnauthorizedAccess"),
    ("StagingAnomaly",        "ITSabotage"),
    ("DataAccessAnomaly",     "ITSabotage"),
    ("AuthAnomaly",           "ITSabotage"),
    ("FlightRiskAnomaly",     "IPTheft"),
    ("ShadowITAnomaly",       "IPTheft"),
    ("DataMovementAnomaly",   "IPTheft"),
    # Layer 3 → Layer 4
    ("DataExfiltration",      "RiskLevel"),
    ("UnauthorizedAccess",    "RiskLevel"),
    ("ITSabotage",            "RiskLevel"),
    ("IPTheft",               "RiskLevel"),
]

# Colours per layer
LAYER_COLORS = {
    "evidence":   "#5dade2",   # cool blue
    "behavioral": "#f39c12",   # amber
    "threat":     "#e74c3c",   # red
    "risk":       "#2ecc71",   # green
}


# ---------------------------------------------------------------------------
# Position helper
# ---------------------------------------------------------------------------

def _layer_positions() -> dict:
    """Assign (x, y) positions for a clean 4-layer hierarchical layout."""
    pos = {}
    layers = [LAYER1_EVIDENCE, LAYER2_BEHAVIORAL, LAYER3_THREATS, LAYER4_RISK]
    y_vals = [0.0, 1.2, 2.4, 3.6]   # bottom → top

    for layer_nodes, y in zip(layers, y_vals):
        n = len(layer_nodes)
        xs = np.linspace(0, 1, n + 2)[1:-1]   # evenly spaced, avoid edges
        for i, (node, x) in enumerate(zip(layer_nodes, xs)):
            # Offset y for dense layers to avoid clutter (zigzag pattern)
            y_offset = 0.0
            if n > 10:
                y_offset = 0.2 if i % 2 == 0 else -0.2
            elif n > 5:
                y_offset = 0.1 if i % 2 == 0 else -0.1
                
            pos[node] = (x, y + y_offset)

    return pos


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_bayesian_network(output_path: str = "bayesian_network.png"):
    """
    Draw the 35-node CERT Insider Threat BN and save to output_path.
    """
    if not (_HAS_MPL and _HAS_NX):
        print(f"  [visualize_bn] matplotlib or networkx not available — skipping plot.")
        return

    G = nx.DiGraph()
    G.add_nodes_from(LAYER1_EVIDENCE + LAYER2_BEHAVIORAL + LAYER3_THREATS + LAYER4_RISK)
    G.add_edges_from(EDGES)

    pos = _layer_positions()

    # Node colours
    node_colors = []
    for node in G.nodes():
        if node in LAYER1_EVIDENCE:
            node_colors.append(LAYER_COLORS["evidence"])
        elif node in LAYER2_BEHAVIORAL:
            node_colors.append(LAYER_COLORS["behavioral"])
        elif node in LAYER3_THREATS:
            node_colors.append(LAYER_COLORS["threat"])
        else:
            node_colors.append(LAYER_COLORS["risk"])

    # Shorten labels for readability
    label_map = {n: n.replace("Activity", "Act.").replace("Anomaly", "Anom.")
                    .replace("Unusual", "Unusual\n").replace("High", "High\n")
                    .replace("Removable", "Removable\n").replace("File", "File\n")
                    .replace("External", "External\n").replace("Communication", "Comms.\n")
                    .replace("Storage", "Storage\n").replace("Machine", "Machine\n")
                    .replace("TimeBased", "Time\nBased").replace("ShadowIT", "Shadow\nIT")
                    .replace("FlightRisk", "Flight\nRisk")
                 for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(34, 18)) # Expanded size for 35 nodes
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color="#888888",
        alpha=0.5,
        width=1.2,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_color=node_colors,
        node_size=2800,
        alpha=0.95,
        edgecolors="white",
        linewidths=2.0,
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        labels=label_map,
        ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=LAYER_COLORS["evidence"],   label="Layer 1: Observable Evidence (21)"),
        mpatches.Patch(color=LAYER_COLORS["behavioral"], label="Layer 2: Behavioral Indicators (9)"),
        mpatches.Patch(color=LAYER_COLORS["threat"],     label="Layer 3: Threat Hypotheses (4)"),
        mpatches.Patch(color=LAYER_COLORS["risk"],       label="Layer 4: Risk Level (1)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              facecolor="#2c2c54", edgecolor="white",
              labelcolor="white", fontsize=10)

    # Layer labels on left
    for label, y in zip(
        ["Layer 1\nObservable Evidence",
         "Layer 2\nBehavioral Indicators",
         "Layer 3\nThreat Hypotheses",
         "Layer 4\nRisk Level"],
        [0.0, 1.2, 2.4, 3.6]
    ):
        ax.text(
            -0.02, y, label, va="center", ha="right",
            fontsize=12, color="#ecf0f1", style="italic",
            transform=ax.transData,
        )

    ax.set_title(
        "CERT R4.2 Insider Threat Bayesian Network (35 Nodes)",
        color="white", fontsize=16, pad=12,
    )
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  BN visualization saved -> {output_path}")

if __name__ == "__main__":
    render_bayesian_network()
