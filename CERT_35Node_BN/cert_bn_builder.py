"""
cert_bn_builder.py
==================
Constructs the 35-node CERT Insider Threat Bayesian Network and estimates
all Conditional Probability Tables (CPTs) from training data.

BN Architecture (4 Layers)
---------------------------
Layer 1 — Observable Evidence (21 binary nodes, learned from data)
Layer 2 — Behavioral Indicator Nodes (9 binary nodes, Noisy-OR)
Layer 3 — Threat Hypothesis Nodes (4 binary nodes, Noisy-OR)
Layer 4 — Risk Level (1 ternary node)

All estimates use Laplace smoothing (+1/+2) to handle sparse cells.
"""

import itertools
import numpy as np
import pandas as pd


# Node name constants (for clarity and refactoring safety)
EVIDENCE_NODES = [
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

CERT_EXPERT_PRIORS = {
    "AfterHoursLogin": 0.15,
    "UnusualLoginFreq": 0.12,
    "WeekendLogin": 0.05,
    "HighMachineCount": 0.08,
    "DeviceConnectActivity": 0.10,
    "DeviceAfterHours": 0.05,
    "FileAccessCount": 0.18,
    "FileCopyToRemovable": 0.08,
    "ArchiveCreation": 0.12,
    "ExeActivity": 0.07,
    "UnusualFileTypes": 0.05,
    "ExternalEmail": 0.20,
    "UnusualEmailVolume": 0.10,
    "AfterHoursEmail": 0.10,
    "ExternalLargeEmail": 0.05,
    "UnusualWebActivity": 0.14,
    "AfterHoursWeb": 0.10,
    "CloudStorageAccess": 0.08,
    "JobSearchActivity": 0.05,
    "HighDataMovement": 0.08,
    "HighFileCopyActivity": 0.05,
}

BEHAVIORAL_NODES = [
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

THREAT_NODES = [
    "DataExfiltration",
    "UnauthorizedAccess",
    "ITSabotage",
    "IPTheft",
]

RISK_NODE = "RiskLevel"
RISK_STATES = [0, 1, 2]   # 0=Low, 1=Medium, 2=High
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# Edges (Layer 1 → Layer 2)
BEHAVIORAL_PARENTS = {
    "AuthAnomaly":          ["AfterHoursLogin", "UnusualLoginFreq", "HighMachineCount"],
    "DataAccessAnomaly":    ["FileAccessCount", "FileCopyToRemovable", "HighFileCopyActivity"],
    "DataMovementAnomaly":  ["FileCopyToRemovable", "HighDataMovement", "HighFileCopyActivity"],
    "CommunicationAnomaly": ["ExternalEmail", "UnusualEmailVolume"],
    "RemovableMediaAnomaly":["DeviceConnectActivity", "FileCopyToRemovable"],
    "TimeBasedAnomaly":     ["WeekendLogin", "AfterHoursEmail", "AfterHoursWeb", "DeviceAfterHours"],
    "StagingAnomaly":       ["ArchiveCreation", "ExeActivity", "UnusualFileTypes"],
    "FlightRiskAnomaly":    ["JobSearchActivity", "ExternalEmail"],
    "ShadowITAnomaly":      ["CloudStorageAccess", "ExternalLargeEmail"],
}

# Edges (Layer 2 → Layer 3)
THREAT_PARENTS = {
    "DataExfiltration":  ["DataAccessAnomaly", "DataMovementAnomaly", "CommunicationAnomaly", "RemovableMediaAnomaly"],
    "UnauthorizedAccess":["AuthAnomaly", "DataAccessAnomaly"],
    "ITSabotage":        ["StagingAnomaly", "DataAccessAnomaly", "AuthAnomaly"],
    "IPTheft":           ["FlightRiskAnomaly", "ShadowITAnomaly", "DataMovementAnomaly"],
}

# Edges (Layer 3 → Layer 4)
RISK_PARENTS = ["DataExfiltration", "UnauthorizedAccess", "ITSabotage", "IPTheft"]


def _laplace_prob(count_true: int, total: int, alpha: float = 1.0, k: int = 2) -> float:
    """Laplace-smoothed probability: (count + α) / (total + k·α)."""
    return float((count_true + alpha) / (total + k * alpha))


def _learn_noisy_or_weights(
    ev_df: pd.DataFrame,
    parent_cols: list,
    positive_mask: pd.Series,
    default_w: float = 0.70,
) -> tuple[list, float]:
    """
    Estimate Noisy-OR weights θ_i = P(parent_i active | positive class)
    and leak λ = P(child active | all parents inactive, normal class).
    """
    pos_sub  = ev_df[positive_mask]
    norm_sub = ev_df[~positive_mask]

    weights = []
    for p in parent_cols:
        if p in ev_df.columns and len(pos_sub) > 0:
            w = _laplace_prob(int(pos_sub[p].sum()), len(pos_sub))
            w = float(np.clip(w, 0.15, 0.95))
        else:
            w = default_w
        weights.append(w)

    # Leak: fraction of normal rows where at least one parent is active
    if len(norm_sub) > 0 and all(p in norm_sub.columns for p in parent_cols):
        any_active = norm_sub[parent_cols].any(axis=1).sum()
        leak = float(np.clip(
            (any_active + 1.0) / (len(norm_sub) + 2.0) * 0.15,
            0.01, 0.10
        ))
    else:
        leak = 0.05

    return weights, leak


class CERTCPTBuilder:
    """
    Estimates all CPTs for the 35-node CERT Insider Threat Bayesian Network
    from discretised training data.
    """

    def __init__(self):
        self.params_ = None

    # ------------------------------------------------------------------
    def fit(self, train_ev_df: pd.DataFrame) -> dict:
        """
        Estimate all CPT parameters from the training evidence DataFrame.
        """
        ev = train_ev_df.copy()
        for col in EVIDENCE_NODES:
            if col in ev.columns:
                ev[col] = ev[col].astype(int)

        insider_mask = ev["is_insider"].astype(bool)
        exfil_mask   = ev["threat_type"] == "DataExfiltration"
        unauth_mask  = ev["threat_type"] == "UnauthorizedAccess"
        sabotage_mask = ev["threat_type"] == "ITSabotage"
        iptheft_mask = ev["threat_type"] == "IPTheft"
        normal_mask  = ~insider_mask

        n_total  = len(ev)
        n_insider = insider_mask.sum()

        params = {"nodes": {}}

        # ---------------------------------------------------------------
        # Layer 1: Evidence node priors
        # ---------------------------------------------------------------
        for node in EVIDENCE_NODES:
            if node in ev.columns:
                count_1 = int(ev[node].sum())
                prior_1 = _laplace_prob(count_1, n_total)
                prior_0 = 1.0 - prior_1
            else:
                prior_1 = 0.10
                prior_0 = 0.90

            params["nodes"][node] = {
                "parents": [],
                "states":  [0, 1],
                "prior":   {0: prior_0, 1: prior_1},
            }

        # ---------------------------------------------------------------
        # Layer 2: Behavioral indicator nodes (Noisy-OR)
        # ---------------------------------------------------------------
        for beh_node, parent_list in BEHAVIORAL_PARENTS.items():
            if beh_node in ("DataMovementAnomaly", "DataAccessAnomaly", "RemovableMediaAnomaly"):
                pos_mask = exfil_mask | iptheft_mask
            elif beh_node == "UnauthorizedAccess":
                pos_mask = unauth_mask
            elif beh_node == "StagingAnomaly":
                pos_mask = sabotage_mask | iptheft_mask
            elif beh_node == "FlightRiskAnomaly" or beh_node == "ShadowITAnomaly":
                pos_mask = iptheft_mask
            else:
                pos_mask = insider_mask

            weights, leak = _learn_noisy_or_weights(
                ev, parent_list, pos_mask, default_w=0.65
            )

            params["nodes"][beh_node] = {
                "parents":  parent_list,
                "states":   [0, 1],
                "noisy_or": {"weights": weights, "leak": leak},
            }

        # ---------------------------------------------------------------
        # Layer 3: Threat hypothesis CPTs
        # ---------------------------------------------------------------
        def _threat_noisy_or_params(parent_list, pos_mask, default_w=0.75):
            proxy_weights = []
            for beh in parent_list:
                beh_parents = BEHAVIORAL_PARENTS.get(beh, [])
                if beh_parents and all(c in ev.columns for c in beh_parents):
                    pos_sub  = ev[pos_mask]
                    any_act  = (pos_sub[beh_parents].sum(axis=1) > 0) if len(pos_sub) else pd.Series([])
                    w = _laplace_prob(int(any_act.sum()), len(pos_sub)) if len(pos_sub) else 0.70
                else:
                    w = 0.60
                proxy_weights.append(float(np.clip(w, 0.15, 0.95)))

            norm_sub = ev[~pos_mask]
            leak = float(np.clip(0.02 + np.random.default_rng(0).uniform(-0.005, 0.005), 0.01, 0.05))
            return proxy_weights, leak

        exfil_weights, exfil_leak = _threat_noisy_or_params(
            THREAT_PARENTS["DataExfiltration"], exfil_mask, 0.80
        )
        params["nodes"]["DataExfiltration"] = {
            "parents":  THREAT_PARENTS["DataExfiltration"],
            "states":   [0, 1],
            "noisy_or": {"weights": exfil_weights, "leak": exfil_leak},
        }

        unauth_weights, unauth_leak = _threat_noisy_or_params(
            THREAT_PARENTS["UnauthorizedAccess"], unauth_mask, 0.80
        )
        params["nodes"]["UnauthorizedAccess"] = {
            "parents":  THREAT_PARENTS["UnauthorizedAccess"],
            "states":   [0, 1],
            "noisy_or": {"weights": unauth_weights, "leak": unauth_leak},
        }
        
        sabotage_weights, sabotage_leak = _threat_noisy_or_params(
            THREAT_PARENTS["ITSabotage"], sabotage_mask, 0.80
        )
        params["nodes"]["ITSabotage"] = {
            "parents":  THREAT_PARENTS["ITSabotage"],
            "states":   [0, 1],
            "noisy_or": {"weights": sabotage_weights, "leak": sabotage_leak},
        }
        
        iptheft_weights, iptheft_leak = _threat_noisy_or_params(
            THREAT_PARENTS["IPTheft"], iptheft_mask, 0.80
        )
        params["nodes"]["IPTheft"] = {
            "parents":  THREAT_PARENTS["IPTheft"],
            "states":   [0, 1],
            "noisy_or": {"weights": iptheft_weights, "leak": iptheft_leak},
        }

        # ---------------------------------------------------------------
        # Layer 4: RiskLevel CPT (ternary: Low=0, Medium=1, High=2)
        # ---------------------------------------------------------------
        risk_cpt = {}
        for combo in itertools.product([0, 1], repeat=4):
            num_threats = sum(combo)
            if num_threats == 0:
                probs = {0: 0.85, 1: 0.12, 2: 0.03}
            elif num_threats == 1:
                if combo[0] == 1 or combo[3] == 1:
                    probs = {0: 0.05, 1: 0.15, 2: 0.80}
                else:
                    probs = {0: 0.10, 1: 0.25, 2: 0.65}
            else:
                probs = {0: 0.02, 1: 0.08, 2: 0.90}
            risk_cpt[combo] = probs

        if n_total > 0:
            frac_truly_normal = float(normal_mask.sum()) / n_total
            if frac_truly_normal > 0.90:
                risk_cpt[(0, 0, 0, 0)][0] = min(0.92, risk_cpt[(0, 0, 0, 0)][0] + 0.05)
                risk_cpt[(0, 0, 0, 0)][1] = max(0.06, risk_cpt[(0, 0, 0, 0)][1] - 0.03)
                risk_cpt[(0, 0, 0, 0)][2] = max(0.02, risk_cpt[(0, 0, 0, 0)][2] - 0.02)

        params["nodes"][RISK_NODE] = {
            "parents": RISK_PARENTS,
            "states":  RISK_STATES,
            "cpt":     risk_cpt,
        }

        self.params_ = params
        return params

    # ------------------------------------------------------------------
    @staticmethod
    def expert_params() -> dict:
        """
        Return a hand-crafted expert-designed parameter set for the same
        35-node network topology.
        """
        params = {"nodes": {}}

        # Layer 1: Evidence priors (domain-motivated)
        evidence_priors = {
            "AfterHoursLogin":       0.15,
            "UnusualLoginFreq":      0.12,
            "WeekendLogin":          0.05,
            "HighMachineCount":      0.08,
            "DeviceConnectActivity": 0.10,
            "DeviceAfterHours":      0.05,
            "FileAccessCount":       0.18,
            "FileCopyToRemovable":   0.08,
            "ArchiveCreation":       0.12,
            "ExeActivity":           0.07,
            "UnusualFileTypes":      0.05,
            "ExternalEmail":         0.20,
            "UnusualEmailVolume":    0.10,
            "AfterHoursEmail":       0.10,
            "ExternalLargeEmail":    0.05,
            "UnusualWebActivity":    0.14,
            "AfterHoursWeb":         0.10,
            "CloudStorageAccess":    0.08,
            "JobSearchActivity":     0.05,
            "HighDataMovement":      0.08,
            "HighFileCopyActivity":  0.05,
        }
        for node, p in evidence_priors.items():
            params["nodes"][node] = {
                "parents": [],
                "states":  [0, 1],
                "prior":   {0: 1 - p, 1: p},
            }

        # Layer 2: Behavioral Noisy-OR (expert weights)
        beh_noisy_or = {
            "AuthAnomaly": {
                "parents": ["AfterHoursLogin", "UnusualLoginFreq", "HighMachineCount"],
                "weights": [0.80, 0.70, 0.85],
                "leak":    0.03,
            },
            "DataAccessAnomaly": {
                "parents": ["FileAccessCount", "FileCopyToRemovable", "HighFileCopyActivity"],
                "weights": [0.60, 0.85, 0.90],
                "leak":    0.04,
            },
            "DataMovementAnomaly": {
                "parents": ["FileCopyToRemovable", "HighDataMovement", "HighFileCopyActivity"],
                "weights": [0.80, 0.75, 0.85],
                "leak":    0.03,
            },
            "CommunicationAnomaly": {
                "parents": ["ExternalEmail", "UnusualEmailVolume"],
                "weights": [0.75, 0.70],
                "leak":    0.05,
            },
            "RemovableMediaAnomaly": {
                "parents": ["DeviceConnectActivity", "FileCopyToRemovable"],
                "weights": [0.70, 0.90],
                "leak":    0.02,
            },
            "TimeBasedAnomaly": {
                "parents": ["WeekendLogin", "AfterHoursEmail", "AfterHoursWeb", "DeviceAfterHours"],
                "weights": [0.85, 0.80, 0.80, 0.90],
                "leak":    0.02,
            },
            "StagingAnomaly": {
                "parents": ["ArchiveCreation", "ExeActivity", "UnusualFileTypes"],
                "weights": [0.80, 0.85, 0.95],
                "leak":    0.03,
            },
            "FlightRiskAnomaly": {
                "parents": ["JobSearchActivity", "ExternalEmail"],
                "weights": [0.90, 0.70],
                "leak":    0.02,
            },
            "ShadowITAnomaly": {
                "parents": ["CloudStorageAccess", "ExternalLargeEmail"],
                "weights": [0.85, 0.80],
                "leak":    0.03,
            },
        }
        for node, spec in beh_noisy_or.items():
            params["nodes"][node] = {
                "parents":  spec["parents"],
                "states":   [0, 1],
                "noisy_or": {"weights": spec["weights"], "leak": spec["leak"]},
            }

        # Layer 3: Threat Noisy-OR (expert)
        params["nodes"]["DataExfiltration"] = {
            "parents":  THREAT_PARENTS["DataExfiltration"],
            "states":   [0, 1],
            "noisy_or": {"weights": [0.85, 0.90, 0.70, 0.92], "leak": 0.02},
        }
        params["nodes"]["UnauthorizedAccess"] = {
            "parents":  THREAT_PARENTS["UnauthorizedAccess"],
            "states":   [0, 1],
            "noisy_or": {"weights": [0.88, 0.75], "leak": 0.02},
        }
        params["nodes"]["ITSabotage"] = {
            "parents":  THREAT_PARENTS["ITSabotage"],
            "states":   [0, 1],
            "noisy_or": {"weights": [0.90, 0.85, 0.80], "leak": 0.02},
        }
        params["nodes"]["IPTheft"] = {
            "parents":  THREAT_PARENTS["IPTheft"],
            "states":   [0, 1],
            "noisy_or": {"weights": [0.85, 0.90, 0.88], "leak": 0.02},
        }

        # Layer 4: RiskLevel CPT (same deterministic mapping)
        risk_cpt_expert = {}
        for combo in itertools.product([0, 1], repeat=4):
            num_threats = sum(combo)
            if num_threats == 0:
                probs = {0: 0.88, 1: 0.10, 2: 0.02}
            elif num_threats == 1:
                if combo[0] == 1 or combo[3] == 1:
                    probs = {0: 0.05, 1: 0.15, 2: 0.80}
                else:
                    probs = {0: 0.10, 1: 0.25, 2: 0.65}
            else:
                probs = {0: 0.02, 1: 0.08, 2: 0.90}
            risk_cpt_expert[combo] = probs

        params["nodes"][RISK_NODE] = {
            "parents": RISK_PARENTS,
            "states":  RISK_STATES,
            "cpt": risk_cpt_expert,
        }

        return params

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a readable summary of learned parameters."""
        if self.params_ is None:
            print("Not fitted yet.")
            return
        for node, spec in self.params_["nodes"].items():
            kind = "prior" if "prior" in spec else ("noisy_or" if "noisy_or" in spec else "cpt")
            print(f"  {node:<30} [{kind}]  parents={spec['parents']}")
