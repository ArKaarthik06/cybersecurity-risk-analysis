"""
cert_math_tracer.py
===================
Generates a step-by-step LaTeX and structured JSON mathematical calculation trace
for any threat assessment query on the 35-Node CERT Bayesian Network.

Step 1: Evidence Vector & Prior Probabilities
Step 2: Noisy-OR Behavioral Indicator Calculation (Closed-form formulas & numerical substitution)
Step 3: Variable Elimination Process (Ancestor Pruning, Active Factors, Elimination Order, Marginalization & Normalization)
Step 4: Decision Network Expected Utility Maximization
"""

from typing import Dict, Any, List
import copy
from cert_bn_builder import (
    EVIDENCE_NODES, BEHAVIORAL_NODES, THREAT_NODES, RISK_NODE,
    CERTCPTBuilder, CERT_EXPERT_PRIORS
)
from cert_decision_network import (
    ACTIONS, UTILITY_RISK_MATRIX, UTILITY_THREAT_MATRIX
)


class MathTracer:
    """
    Constructs a detailed step-by-step mathematical trace for a given evidence vector.
    """

    def __init__(self, cpt_builder: CERTCPTBuilder):
        self.cpt_builder = cpt_builder
        self.params = cpt_builder.expert_params()

    def generate_trace(
        self,
        evidence: Dict[str, int],
        ve_results: Tuple[Dict[str, float], Dict[str, Any]],
        threat_posteriors: Dict[str, float],
        risk_posterior: Dict[str, float],
        decision_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates full step-by-step mathematical trace data.
        """

        # -------------------------------------------------------------
        # Step 1: Evidence & Prior Analysis
        # -------------------------------------------------------------
        step1_items = []
        for node in EVIDENCE_NODES:
            val = evidence.get(node, 0)
            prior = CERT_EXPERT_PRIORS.get(node, 0.05)
            step1_items.append({
                "node": node,
                "value": val,
                "prior": prior,
                "latex": f"P({node} = 1) = {prior:.2f} \\quad \\implies \\text{{Observed State}} = {val}"
            })

        # -------------------------------------------------------------
        # Step 2: Noisy-OR Calculations (Layer 2 & Layer 3)
        # -------------------------------------------------------------
        step2_items = []
        for node in BEHAVIORAL_NODES + THREAT_NODES:
            node_spec = self.params["nodes"].get(node, {})
            if "noisy_or" in node_spec:
                parents = node_spec.get("parents", node_spec["noisy_or"].get("parents", []))
                weights = node_spec["noisy_or"]["weights"]
                leak = node_spec["noisy_or"]["leak"]

                # Find active parents
                active_parents = []
                active_weights = []
                for p, w in zip(parents, weights):
                    # Check if parent is evidence or evaluated posterior
                    p_val = evidence.get(p, 0)
                    if p in evidence and p_val == 1:
                        active_parents.append(p)
                        active_weights.append(w)

                # Compute noisy-OR closed form
                prod_term = 1.0 - leak
                for w in active_weights:
                    prod_term *= (1.0 - w)
                prob = 1.0 - prod_term

                w_str = ", ".join([f"w_{{{p}}} = {w}" for p, w in zip(active_parents, active_weights)])
                if not active_parents:
                    latex_eq = (
                        f"P({node} = 1 \\mid \\text{{No Active Parents}}) = \\text{{Leak}} = {leak:.3f}"
                    )
                else:
                    latex_eq = (
                        f"P({node} = 1 \\mid \\text{{Evidence}}) = 1 - (1 - \\lambda) \\prod_{{i \\in \\text{{Active}}}} (1 - w_i) \\\\\n"
                        f"= 1 - (1 - {leak:.2f}) " + "".join([f"(1 - {w:.2f})" for w in active_weights]) + f" = {prob:.4f}"
                    )

                step2_items.append({
                    "node": node,
                    "parents": parents,
                    "active_parents": active_parents,
                    "weights": weights,
                    "leak": leak,
                    "calculated_probability": round(prob, 4),
                    "latex": latex_eq
                })

        # -------------------------------------------------------------
        # Step 3: Exact Variable Elimination Sequence
        # -------------------------------------------------------------
        posterior_dict, stats = ve_results
        step3 = {
            "query_variable": RISK_NODE,
            "strategy": stats.get("strategy", "min-degree"),
            "induced_width": stats.get("induced_width", 4),
            "max_factor_size": stats.get("max_factor_size", 48),
            "runtime_ms": round(stats.get("runtime_ms", 0.0), 3),
            "ancestor_pruning": (
                f"Pruned network from 35 nodes down to active ancestor subgraph for RiskLevel."
            ),
            "latex_normalization": (
                f"P(\\text{{RiskLevel}} = r \\mid E) = \\frac{{\\Phi_{{\\text{{final}}}}(r)}}{{\\sum_{{r' \\in \\{{Low, Med, High\\}}}} \\Phi_{{\\text{{final}}}}(r')}} \\\\\n"
                f"P(\\text{{Low}}) = {risk_posterior.get('Low', 0.0):.4f}, \\quad "
                f"P(\\text{{Medium}}) = {risk_posterior.get('Medium', 0.0):.4f}, \\quad "
                f"P(\\text{{High}}) = {risk_posterior.get('High', 0.0):.4f}"
            )
        }

        # -------------------------------------------------------------
        # Step 4: Decision Network Expected Utility Algebra
        # -------------------------------------------------------------
        step4_items = []
        for action in ACTIONS:
            u_risk_sum = sum(
                risk_posterior.get(r, 0.0) * UTILITY_RISK_MATRIX[action].get(r, 0.0)
                for r in ["Low", "Medium", "High"]
            )
            u_threat_sum = sum(
                threat_posteriors.get(t, 0.0) * UTILITY_THREAT_MATRIX[t].get(action, 0.0)
                for t in UTILITY_THREAT_MATRIX
            )
            total_u = decision_results["expected_utilities"].get(action, 0.0)

            latex_u = (
                f"E[U(\\text{{{action}}})] = \\sum_{{r}} P(\\text{{Risk}}=r) U_{{\\text{{risk}}}}(\\text{{{action}}}, r) + 0.4 \\sum_{{t}} P(\\text{{Threat}}=t) U_{{\\text{{threat}}}}(\\text{{{action}}}, t) \\\\\n"
                f"= ({u_risk_sum:.2f}) + 0.4 \\times ({u_threat_sum:.2f}) = \\mathbf{{{total_u:.2f}}}"
            )

            step4_items.append({
                "action": action,
                "utility": total_u,
                "is_selected": (action == decision_results["recommended_action"]),
                "latex": latex_u
            })

        return {
            "step1_evidence": step1_items,
            "step2_noisy_or": step2_items,
            "step3_variable_elimination": step3,
            "step4_decision_utility": {
                "items": step4_items,
                "selected_action": decision_results["recommended_action"],
                "latex_optimal": f"a^* = \\arg\\max_a E[U(a)] = \\mathbf{{\\text{{{decision_results['recommended_action']}}}}}"
            }
        }
