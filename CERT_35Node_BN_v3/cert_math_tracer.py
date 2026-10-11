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

from typing import Dict, Any, List, Tuple
import copy
from cert_bn_builder import EVIDENCE_NODES, BEHAVIORAL_NODES, THREAT_NODES, RISK_NODE
from cert_decision_network import ACTIONS, ACTION_COST, THREAT_LOSS, RESIDUAL


class MathTracer:
    """
    Constructs a detailed step-by-step mathematical trace for a given evidence vector.
    """

    def __init__(self, params: dict):
        # the SAME fitted parameters the inference engine uses (v3: no separate expert copy)
        self.params = params

    def generate_trace(
        self,
        evidence: Dict[str, int],
        ve_results: Tuple[Dict[str, float], Dict[str, Any]],
        threat_posteriors: Dict[str, float],
        risk_posterior: Dict[str, float],
        decision_results: Dict[str, Any],
        behavioral_posteriors: Dict[str, float] = None,
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
            prior = self.params["nodes"][node]["prior"][1]
            step1_items.append({
                "node": node,
                "value": val,
                "prior": prior,
                "latex": f"P({node} = 1) = {prior:.4f} \\quad \\implies \\text{{Observed State}} = {val}"
            })

        # -------------------------------------------------------------
        # Step 2: Noisy-OR Calculations (Layer 2 & Layer 3)
        # -------------------------------------------------------------
        step2_items = []
        parent_post = behavioral_posteriors
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
                    elif p not in evidence and parent_post is not None and parent_post.get(p, 0) >= 0.5:
                        active_parents.append(p)        # latent parent: shown as active when its posterior >= 0.5
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
                    mult_str = "".join([f"(1 - {w:.2f})" for w in active_weights])
                    latex_eq = (
                        f"\\begin{{aligned}}\n"
                        f"P({node} = 1 \\mid \\text{{Evidence}}) &= 1 - (1 - \\lambda) \\prod_{{i \\in \\text{{Active}}}} (1 - w_i) \\\\\n"
                        f"&= 1 - (1 - {leak:.2f}) {mult_str} \\\\\n"
                        f"&= \\mathbf{{{prob:.4f}}}\n"
                        f"\\end{{aligned}}"
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
            "runtime_ms": round(stats.get("runtime_sec", 0.0) * 1000, 3),
            "ancestor_pruning": (
                "Pruned the 35-node DAG to the active ancestor subgraph of RiskLevel."
            ),
            "latex_normalization": (
                f"\\begin{{aligned}}\n"
                f"P(R=r \\mid \\mathbf{{E}}) &= "
                f"\\frac{{\\Phi_{{\\text{{final}}}}(r)}}{{\\displaystyle\\sum_{{r'}} \\Phi_{{\\text{{final}}}}(r')}} \\\\[6pt]\n"
                f"P(R=\\text{{Low}} \\mid \\mathbf{{E}}) &= \\mathbf{{{risk_posterior.get('Low', 0.0):.4f}}} \\\\\n"
                f"P(R=\\text{{Med}} \\mid \\mathbf{{E}}) &= \\mathbf{{{risk_posterior.get('Medium', 0.0):.4f}}} \\\\\n"
                f"P(R=\\text{{High}} \\mid \\mathbf{{E}}) &= \\mathbf{{{risk_posterior.get('High', 0.0):.4f}}}\n"
                f"\\end{{aligned}}"
            )
        }

        # -------------------------------------------------------------
        # Step 4: Decision Network Expected Utility Algebra
        # -------------------------------------------------------------
        step4_items = []
        for action in ACTIONS:
            cost = ACTION_COST[action]
            exp_loss = sum(threat_posteriors.get(t, 0.0) * THREAT_LOSS[t] * RESIDUAL[action][t] for t in THREAT_LOSS)
            total_u = decision_results["expected_utilities"].get(action, 0.0)
            action_latex = action.replace("_", "\\_")
            latex_u = (
                f"\\begin{{aligned}}\n"
                f"E[U(\\texttt{{{action_latex}}})] &= "
                f"-C(a) - \\underbrace{{\\sum_{{t}} P(T_t \\mid \\mathbf{{E}}) \\cdot L_t \\cdot r_t(a)}}_{{{exp_loss:.3f}}} \\\\\n"
                f"&= -{cost:.3f} - {exp_loss:.3f} \\\\\n"
                f"&= \\boxed{{\\mathbf{{{total_u:.3f}}}}}\n"
                f"\\end{{aligned}}"
            )
            step4_items.append({"action": action, "utility": total_u,
                                "is_selected": (action == decision_results["recommended_action"]),
                                "latex": latex_u})

        # -------------------------------------------------------------
        # Step 5: Final Risk Probability & Score Derivation
        # -------------------------------------------------------------
        p_low = risk_posterior.get("Low", 0.0)
        p_med = risk_posterior.get("Medium", 0.0)
        p_high = risk_posterior.get("High", 0.0)
        p_elevated = p_med + p_high
        risk_score = 0.0 * p_low + 0.5 * p_med + 1.0 * p_high

        if p_high >= 0.50:
            classification = "HIGH RISK"
        elif p_elevated >= 0.40:
            classification = "MEDIUM RISK"
        else:
            classification = "LOW RISK"

        latex_final_risk = (
            f"\\begin{{aligned}}\n"
            f"P(\\text{{High Risk}}) &= {p_high:.4f} \\quad ({(p_high * 100):.1f}\\%)\\\\\n"
            f"P(\\text{{Elevated Risk}}) &= P(\\text{{Medium}}) + P(\\text{{High}}) = {p_med:.4f} + {p_high:.4f} = \\mathbf{{{p_elevated:.4f}}} \\quad ({(p_elevated * 100):.1f}\\%)\\\\\n"
            f"E[\\text{{Risk Score}}] &= 0.0 \\times P(\\text{{Low}}) + 0.5 \\times P(\\text{{Medium}}) + 1.0 \\times P(\\text{{High}})\\\\\n"
            f"&= 0.0 \\times {p_low:.4f} + 0.5 \\times {p_med:.4f} + 1.0 \\times {p_high:.4f} = \\mathbf{{{risk_score:.4f}}} \\quad ({(risk_score * 100):.1f}\\%)\\\\\n"
            f"\\text{{Final Assessment}} &= \\mathbf{{\\text{{{classification}}}}}\n"
            f"\\end{{aligned}}"
        )

        step5 = {
            "p_low": round(p_low, 4),
            "p_medium": round(p_med, 4),
            "p_high": round(p_high, 4),
            "p_elevated": round(p_elevated, 4),
            "risk_score": round(risk_score, 4),
            "classification": classification,
            "latex": latex_final_risk
        }

        return {
            "step1_evidence": step1_items,
            "step2_noisy_or": step2_items,
            "step3_variable_elimination": step3,
            "step4_decision_utility": {
                "items": step4_items,
                "selected_action": decision_results["recommended_action"],
                "latex_optimal": (
                    f"a^* = \\arg\\max_{{a \\in \\mathcal{{A}}}} \\; E[U(a)] = "
                    f"\\texttt{{{decision_results['recommended_action'].replace('_', '\\_')}}}"
                )
            },
            "step5_final_risk": step5
        }

