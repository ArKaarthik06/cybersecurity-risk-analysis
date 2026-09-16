"""
cert_decision_network.py
========================
Decision Network Extension for the 35-Node CERT Bayesian Network.

Extends the probabilistic threat assessment with a decision node
(Recommended Action) and an explicit Expected Utility Maximization Engine:
  a* = argmax_a E[U(a)]

Actions:
--------
  1. MONITOR               : Log & monitor activity (Low risk baseline)
  2. REQUEST_MFA           : Step-up authentication / challenge user (Medium risk)
  3. RESTRICT_PRIVILEGES   : Revoke USB / Cloud access / restrict file permissions (High risk exfiltration/theft)
  4. LOCK_ACCOUNT_ISOLATE  : Immediate account lockout & host network isolation (Critical/Sabotage/Mass exfiltration)

Utility Matrix:
---------------
  Balances operational interruption costs (False Positive costs) vs
  catastrophic security breach damage (Missed Attack costs).
"""

from typing import Dict, Any, List, Tuple

# Action Definitions
ACTIONS = [
    "MONITOR",
    "REQUEST_MFA",
    "RESTRICT_PRIVILEGES",
    "LOCK_ACCOUNT_ISOLATE"
]

ACTION_DESCRIPTIONS = {
    "MONITOR": "Continue standard logging and passive behavioral monitoring. No user disruption.",
    "REQUEST_MFA": "Issue a step-up authentication challenge (MFA) to verify user identity.",
    "RESTRICT_PRIVILEGES": "Temporarily restrict removable media, external email attachments, and cloud storage uploads.",
    "LOCK_ACCOUNT_ISOLATE": "Immediately lock user credentials, terminate active sessions, and isolate workstation from the network."
}

# Utility matrix over Risk Levels: U(Action, RiskLevel)
# Scale: -100 (catastrophic) to 100 (optimal match)
UTILITY_RISK_MATRIX = {
    "MONITOR": {
        "Low": 100,      # Perfect decision
        "Medium": 20,    # Slightly under-reacting
        "High": -90      # Severe under-reaction (missed high risk)
    },
    "REQUEST_MFA": {
        "Low": 40,       # Minor user friction cost (-10)
        "Medium": 90,    # Great match for medium suspicion
        "High": 10       # Under-reacting to active attack
    },
    "RESTRICT_PRIVILEGES": {
        "Low": -20,      # False alarm productivity cost
        "Medium": 60,    # Good containment
        "High": 95       # Excellent containment of data theft
    },
    "LOCK_ACCOUNT_ISOLATE": {
        "Low": -80,      # Major false alarm disruption cost
        "Medium": -10,   # Heavy-handed response
        "High": 100      # Essential for critical breach prevention
    }
}

# Additional Utility weights based on specific Threat Hypotheses: U(Action, Threat)
UTILITY_THREAT_MATRIX = {
    "DataExfiltration": {
        "MONITOR": -100,
        "REQUEST_MFA": -20,
        "RESTRICT_PRIVILEGES": 100,
        "LOCK_ACCOUNT_ISOLATE": 70
    },
    "UnauthorizedAccess": {
        "MONITOR": -90,
        "REQUEST_MFA": 95,
        "RESTRICT_PRIVILEGES": 40,
        "LOCK_ACCOUNT_ISOLATE": 80
    },
    "ITSabotage": {
        "MONITOR": -100,
        "REQUEST_MFA": -50,
        "RESTRICT_PRIVILEGES": 30,
        "LOCK_ACCOUNT_ISOLATE": 100
    },
    "IPTheft": {
        "MONITOR": -80,
        "REQUEST_MFA": 10,
        "RESTRICT_PRIVILEGES": 95,
        "LOCK_ACCOUNT_ISOLATE": 75
    }
}


class DecisionNetworkEngine:
    """
    Evaluates Expected Utility over Risk Level and Threat Hypotheses posteriors
    to output optimal security response actions.
    """

    def __init__(self):
        self.actions = ACTIONS
        self.descriptions = ACTION_DESCRIPTIONS

    def compute_expected_utility(
        self,
        risk_posterior: Dict[str, float],
        threat_posteriors: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Computes expected utility E[U(a)] for each candidate action.

        E[U(a)] = sum_{r} P(Risk=r) * U_risk(a, r)
                + 0.4 * sum_{t} P(Threat=t) * U_threat(a, t)
        """
        utilities = {}

        for action in self.actions:
            # Risk component
            u_risk = sum(
                risk_posterior.get(r_level, 0.0) * UTILITY_RISK_MATRIX[action].get(r_level, 0.0)
                for r_level in ["Low", "Medium", "High"]
            )

            # Threat component
            u_threat = sum(
                threat_posteriors.get(t_name, 0.0) * UTILITY_THREAT_MATRIX[t_name].get(action, 0.0)
                for t_name in UTILITY_THREAT_MATRIX
            )

            # Total combined expected utility
            total_u = u_risk + 0.4 * u_threat
            utilities[action] = round(total_u, 2)

        # Select action with maximum expected utility
        best_action = max(utilities, key=utilities.get)

        return {
            "expected_utilities": utilities,
            "recommended_action": best_action,
            "action_description": self.descriptions[best_action],
            "utility_breakdown": {
                action: {
                    "expected_utility": utilities[action],
                    "description": self.descriptions[action]
                }
                for action in self.actions
            }
        }
