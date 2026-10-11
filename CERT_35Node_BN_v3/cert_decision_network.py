"""
cert_decision_network.py (v3)
Decision network on top of the 35-node BN.

    E[U(a)] = - C(a) - sum_j  P(T_j | evidence) * L_j * r_j(a)          a* = argmax_a E[U(a)]

* P(T_j | evidence) are the exact VE posteriors of the four threat nodes (no RiskLevel term and no
  0.4 blend, so nothing is double counted).
* C(a)   cost of taking action a (user friction / analyst time), L_j damage if threat j is real and
  unmitigated, r_j(a) fraction of that damage that remains after action a.  All numbers are in
  one common unit ("cost units") and are ASSUMPTIONS to be set by the organisation - they live in
  one table below so they can be audited and varied (see break_even / sensitivity).
"""
from typing import Dict
ACTIONS = ["MONITOR", "REQUEST_MFA", "RESTRICT_PRIVILEGES", "LOCK_ACCOUNT_ISOLATE"]
DESCRIPTIONS = {
    "MONITOR": "Continue standard logging and passive monitoring.",
    "REQUEST_MFA": "Step-up authentication challenge.",
    "RESTRICT_PRIVILEGES": "Restrict removable media, external attachments and cloud uploads.",
    "LOCK_ACCOUNT_ISOLATE": "Lock credentials, end sessions, isolate the workstation.",
}
ACTION_COST = {"MONITOR": 0.0, "REQUEST_MFA": 1.0, "RESTRICT_PRIVILEGES": 5.0, "LOCK_ACCOUNT_ISOLATE": 25.0}
THREAT_LOSS = {"DataExfiltration": 400.0, "UnauthorizedAccess": 200.0, "ITSabotage": 600.0, "IPTheft": 500.0}
RESIDUAL = {   # fraction of the loss that remains after the action
    "MONITOR":              {"DataExfiltration": 1.0, "UnauthorizedAccess": 1.0, "ITSabotage": 1.0, "IPTheft": 1.0},
    "REQUEST_MFA":          {"DataExfiltration": 0.8, "UnauthorizedAccess": 0.3, "ITSabotage": 0.8, "IPTheft": 0.8},
    "RESTRICT_PRIVILEGES":  {"DataExfiltration": 0.2, "UnauthorizedAccess": 0.5, "ITSabotage": 0.6, "IPTheft": 0.2},
    "LOCK_ACCOUNT_ISOLATE": {"DataExfiltration": 0.05, "UnauthorizedAccess": 0.05, "ITSabotage": 0.05, "IPTheft": 0.05},
}

class DecisionNetworkEngine:
    def __init__(self, cost=None, loss=None, residual=None):
        self.cost = cost or ACTION_COST; self.loss = loss or THREAT_LOSS; self.res = residual or RESIDUAL
        self.actions = ACTIONS; self.descriptions = DESCRIPTIONS

    def expected_utility(self, threat_post: Dict[str, float]) -> Dict[str, float]:
        return {a: -self.cost[a] - sum(threat_post.get(t, 0.0) * L * self.res[a][t] for t, L in self.loss.items())
                for a in self.actions}

    def compute_expected_utility(self, risk_posterior, threat_posteriors):
        u = self.expected_utility(threat_posteriors)
        best = max(u, key=u.get)
        return {"expected_utilities": {a: round(v, 3) for a, v in u.items()}, "recommended_action": best,
                "action_description": self.descriptions[best],
                "utility_breakdown": {a: {"expected_utility": round(u[a], 3), "description": self.descriptions[a]} for a in self.actions}}

    def break_even(self, threat: str, a: str = "MONITOR", b: str = "REQUEST_MFA"):
        """P(threat) at which a and b have equal E[U] if only `threat` is active-possible."""
        da = self.loss[threat] * (self.res[a][threat] - self.res[b][threat])
        return (self.cost[b] - self.cost[a]) / da if da > 0 else float("inf")
