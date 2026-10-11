"""
cert_app_server.py
==================
Flask Web Server for the 35-Node Probabilistic Insider Threat System.

Provides REST API endpoints for:
  - Interactive Bayesian Network Schema & Topology (/api/schema)
  - Pre-loaded User-Day Log Samples (/api/samples)
  - Real-time Log Evidence Extraction & BN Risk Analysis (/api/analyze)
  - Step-by-Step Mathematical Trace Generation (/api/math-trace)

Serves the front-end Antigravity IDE-style web dashboard from /static.
"""

import os
import sys
import json
import traceback
import requests
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

# Project imports
from cert_preprocessing import EVIDENCE_NODES
from cert_bn_builder import BEHAVIORAL_NODES, THREAT_NODES, RISK_NODE, RISK_LABELS
from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine
from cert_decision_network import DecisionNetworkEngine
from cert_reporter import CERTReporter
from cert_math_tracer import MathTracer

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="")


# ---------------------------------------------------------------------------
# Load API Keys from keys.env
# ---------------------------------------------------------------------------
_env_path = os.path.join(BASE, "results", "keys.env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    print("[Server] Loaded environment keys from results/keys.env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


# ---------------------------------------------------------------------------
# Global Initialization of Model Components
# ---------------------------------------------------------------------------
print("[Server] Initializing Bayesian Network and Inference Engine...")
def _load_params(path):
    """Load the learned parameters written by build_model.py (JSON -> engine format)."""
    raw = json.load(open(path))
    out = {"nodes": {}}
    for n, spec in raw["nodes"].items():
        spec = dict(spec)
        if "prior" in spec:
            spec["prior"] = {int(k): v for k, v in spec["prior"].items()}
        if "cpt" in spec:
            spec["cpt"] = {tuple(int(x) for x in k.split(",")): {int(a): b for a, b in v.items()}
                           for k, v in spec["cpt"].items()}
        out["nodes"][n] = spec
    return out

_params_path = os.path.join(BASE, "data", "model_params.json")
if not os.path.exists(_params_path):
    sys.exit("data/model_params.json not found - run  python build_model.py  first.")
bn_params = _load_params(_params_path)
print("[Server] Learned BN parameters loaded from data/model_params.json")

bn = BayesianNetwork()
factors = bn.build_factors(bn_params)
ve_engine = VariableEliminationEngine(bn, factors)
decision_engine = DecisionNetworkEngine()
reporter = CERTReporter(ve_engine)
math_tracer = MathTracer(bn_params)
print("[Server] Bayesian Network Engine initialized successfully.")


# ---------------------------------------------------------------------------
# Route: Static File Serving
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Route: Network Schema & Graph Topology
# ---------------------------------------------------------------------------
@app.route("/api/schema", methods=["GET"])
def get_schema():
    """
    Returns the graph nodes, edges, layers, and CPT specifications.
    """
    nodes_info = []
    
    # Layer 1: Evidence
    for node in EVIDENCE_NODES:
        prior_val = bn_params["nodes"][node]["prior"][1]
        nodes_info.append({
            "id": node,
            "label": node,
            "layer": 1,
            "layer_name": "Observable Evidence",
            "type": "evidence",
            "parents": [],
            "prior": prior_val
        })
        
    # Layer 2: Behavioral Indicators
    for node in BEHAVIORAL_NODES:
        spec = bn_params["nodes"].get(node, {})
        noisy_or = spec.get("noisy_or", {})
        parents = spec.get("parents", noisy_or.get("parents", []))
        weights = noisy_or.get("weights", [])
        leak = noisy_or.get("leak", 0.03)
        nodes_info.append({
            "id": node,
            "label": node,
            "layer": 2,
            "layer_name": "Behavioral Indicators",
            "type": "indicator",
            "parents": parents,
            "weights": weights,
            "leak": leak
        })
        
    # Layer 3: Threat Hypotheses
    for node in THREAT_NODES:
        spec = bn_params["nodes"].get(node, {})
        noisy_or = spec.get("noisy_or", {})
        parents = spec.get("parents", noisy_or.get("parents", []))
        weights = noisy_or.get("weights", [])
        leak = noisy_or.get("leak", 0.02)
        nodes_info.append({
            "id": node,
            "label": node,
            "layer": 3,
            "layer_name": "Threat Hypotheses",
            "type": "threat",
            "parents": parents,
            "weights": weights,
            "leak": leak
        })
        
    # Layer 4: Overall Risk
    spec = bn_params["nodes"].get(RISK_NODE, {})
    parents = spec.get("parents", THREAT_NODES)
    raw_cpt = spec.get("cpt", {})
    # Format CPT dictionary keys as string representations for JSON output
    formatted_cpt = {}
    if isinstance(raw_cpt, dict):
        for k, v in raw_cpt.items():
            key_str = str(k) if isinstance(k, tuple) else k
            formatted_cpt[key_str] = {str(a): b for a, b in v.items()}

    nodes_info.append({
        "id": RISK_NODE,
        "label": RISK_NODE,
        "layer": 4,
        "layer_name": "Risk Assessment",
        "type": "risk",
        "parents": parents,
        "cpt": formatted_cpt
    })

    # Collect Edges
    edges = []
    for n in nodes_info:
        for p in n["parents"]:
            edges.append({"from": p, "to": n["id"]})

    return jsonify({
        "nodes": nodes_info,
        "edges": edges,
        "total_nodes": len(nodes_info),
        "total_edges": len(edges)
    })



# ---------------------------------------------------------------------------
# Route: Pre-loaded Samples
# ---------------------------------------------------------------------------
@app.route("/api/samples", methods=["GET"])
def get_samples():
    """
    Returns demo user-days (scored out-of-fold) written by build_model.py.
    """
    sp = os.path.join(BASE, "data", "samples.json")
    samples = json.load(open(sp))["samples"] if os.path.exists(sp) else []

    return jsonify({"samples": samples})


# ---------------------------------------------------------------------------
# Route: Main Threat Analysis Endpoint
# ---------------------------------------------------------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint: Accepts evidence dictionary or sample selection.
    Runs Variable Elimination, Decision Network, Evidence Ablation,
    LLM Report Synthesis, and Step-by-Step Math Trace.
    """
    try:
        data = request.json or {}
        evidence = data.get("evidence", {})
        user_name = data.get("user", "ANALYST_QUERY")
        day_str = data.get("day", "2026-09-16")
        
        # Ensure proper integer types for evidence; unselected checkboxes mean 0 (observed false)
        clean_evidence = {}
        for k in EVIDENCE_NODES:
            clean_evidence[k] = 1 if int(evidence.get(k, 0)) == 1 else 0

        # 1. Exact Variable Elimination for RiskLevel
        risk_dist, stats = ve_engine.query(RISK_NODE, clean_evidence, strategy="min-degree")
        risk_posterior = {
            "Low": float(risk_dist.get(0, 0.0)),
            "Medium": float(risk_dist.get(1, 0.0)),
            "High": float(risk_dist.get(2, 0.0))
        }

        # Determine overall risk level
        if risk_posterior["High"] >= 0.50:
            level_str = "High"
        elif risk_posterior["High"] + risk_posterior["Medium"] >= 0.40:
            level_str = "Medium"
        else:
            level_str = "Low"

        # 2. Query individual Threat Hypotheses
        threat_posteriors = {}
        for threat_node in THREAT_NODES:
            t_dist, _ = ve_engine.query(threat_node, clean_evidence, strategy="min-degree")
            threat_posteriors[threat_node] = float(t_dist.get(1, 0.0))

        # 3. Query Intermediate Behavioral Indicators
        behavioral_posteriors = {}
        for beh_node in BEHAVIORAL_NODES:
            b_dist, _ = ve_engine.query(beh_node, clean_evidence, strategy="min-degree")
            behavioral_posteriors[beh_node] = float(b_dist.get(1, 0.0))

        # 4. Decision Network Expected Utility Maximization
        decision_results = decision_engine.compute_expected_utility(risk_posterior, threat_posteriors)

        # 5. Evidence Sensitivity Ablation (Delta P)
        ablation_results = []
        p_high_base = risk_posterior["High"]
        for ev_node in clean_evidence:
            reduced_ev = {k: v for k, v in clean_evidence.items() if k != ev_node}
            r_dist, _ = ve_engine.query(RISK_NODE, reduced_ev, strategy="min-degree")
            p_high_without = float(r_dist.get(2, 0.0))
            delta_p = p_high_base - p_high_without
            ablation_results.append({
                "node": ev_node,
                "p_high_without": round(p_high_without, 4),
                "delta_p_high": round(delta_p, 4)
            })
        ablation_results.sort(key=lambda x: x["delta_p_high"], reverse=True)

        # 6. Trace Active Threat Paths for Visual Highlighting
        active_threat_paths = _compute_active_paths(clean_evidence, behavioral_posteriors, threat_posteriors, risk_posterior)

        # 7. LLM Security Report Synthesis
        rep_dict, report_text = reporter.generate_and_explain(clean_evidence, user=user_name, day=day_str)

        # 8. Generate Step-by-Step Mathematical Calculation Trace
        math_trace = math_tracer.generate_trace(
            evidence=clean_evidence,
            ve_results=(risk_dist, stats),
            threat_posteriors=threat_posteriors,
            risk_posterior=risk_posterior,
            decision_results=decision_results,
            behavioral_posteriors=behavioral_posteriors
        )

        return jsonify({
            "status": "success",
            "user": user_name,
            "day": day_str,
            "active_evidence": [k for k, v in clean_evidence.items() if v == 1],
            "risk_assessment": {
                "level": level_str,
                "posterior": risk_posterior,
                "p_high": risk_posterior["High"]
            },
            "threat_posteriors": threat_posteriors,
            "top_threat": max(threat_posteriors, key=threat_posteriors.get),
            "behavioral_indicators": behavioral_posteriors,
            "decision_network": decision_results,
            "evidence_ablation": ablation_results,
            "active_threat_paths": active_threat_paths,
            "llm_report": report_text,
            "math_trace": math_trace,
            "inference_stats": {
                "strategy": stats.get("strategy", "min-degree"),
                "induced_width": stats.get("induced_width", 4),
                "max_factor_size": stats.get("max_factor_size", 48),
                "runtime_ms": round(stats.get("runtime_sec", 0.0) * 1000, 3)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


def _compute_active_paths(
    evidence: Dict[str, int],
    behavioral: Dict[str, float],
    threats: Dict[str, float],
    risk: Dict[str, float]
) -> Dict[str, Any]:
    """
    Computes node and edge activation intensity (0.0 to 1.0) for DAG trace highlighting.
    Higher activation → brighter highlight in the DAG visualization.
    """
    node_activations = {}

    # Layer 1: Evidence nodes — binary 1.0 if observed, 0.05 baseline
    for n in EVIDENCE_NODES:
        node_activations[n] = 1.0 if evidence.get(n, 0) == 1 else 0.05

    # Layer 2: Behavioral indicators — scale up for visual clarity
    for n, p in behavioral.items():
        node_activations[n] = round(min(1.0, max(0.05, p * 1.6)), 3)

    # Layer 3: Threat hypotheses — amplified for visibility
    for n, p in threats.items():
        node_activations[n] = round(min(1.0, max(0.05, p * 2.0)), 3)

    # Layer 4: Risk node — use P(High) directly
    node_activations[RISK_NODE] = round(max(0.05, risk.get("High", 0.05)), 3)

    return {
        "node_activations": node_activations,
        "active_evidence_count": len([k for k, v in evidence.items() if v == 1])
    }


# ---------------------------------------------------------------------------
# Route: LLM Analysis via OpenRouter (SSE Streaming)
# ---------------------------------------------------------------------------
@app.route("/api/llm-analyze", methods=["POST"])
def llm_analyze():
    """
    Accepts a prompt string and streams an OpenRouter LLM response as SSE.
    Tries openai/gpt-4o, then openai/gpt-4o-mini, and falls back to a template if needed.
    """
    data   = request.json or {}
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    def stream_fallback():
        template = _build_fallback_analysis(prompt)
        words = template.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            event = json.dumps({"choices": [{"delta": {"content": chunk}}]})
            yield f"data: {event}\n\n"
        yield "data: [DONE]\n\n"

    if not OPENROUTER_API_KEY:
        return Response(stream_with_context(stream_fallback()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def openrouter_stream():
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "http://localhost:5000",
            "X-Title":       "CERT Threat Analyzer"
        }
        
        models_to_try = [
            ("openai/gpt-4o", 1000),
            ("openai/gpt-4o-mini", 1000)
        ]

        stream_success = False

        for model_name, max_tok in models_to_try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert cybersecurity analyst specializing in probabilistic "
                            "insider threat detection using Bayesian Networks. You provide rigorous, "
                            "technically precise, and actionable analysis grounded ONLY in the "
                            "probabilistic data provided. Never invent evidence or change probabilities."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "temperature": 0.2,
                "max_tokens": max_tok
            }

            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=45
                )

                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode("utf-8", errors="ignore")
                            if decoded.startswith("data: "):
                                yield decoded + "\n\n"
                    stream_success = True
                    break
                else:
                    print(f"[Server] OpenRouter model '{model_name}' returned status {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                print(f"[Server] OpenRouter model '{model_name}' exception: {exc}")

        if not stream_success:
            print("[Server] OpenRouter API unavailable or failed. Streaming structured probabilistic analysis fallback.")
            for chunk_event in stream_fallback():
                yield chunk_event
        else:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(openrouter_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )



def _build_fallback_analysis(prompt: str) -> str:
    """
    Structured fallback analysis when no LLM API key is configured.
    Extracts key values from the prompt via string parsing.
    """
    # Extract risk level from prompt
    risk_level = "LOW"
    if "HIGH RISK" in prompt: risk_level = "HIGH"
    elif "MEDIUM RISK" in prompt: risk_level = "MEDIUM"

    return f"""## Executive Risk Summary

The 35-Node Bayesian Network has completed Variable Elimination inference and assessed this user-day as **{risk_level} RISK**. The posterior probabilities reflect the cumulative causal influence of observed evidence signals propagating through the four-layer network: Observable Evidence → Behavioral Indicators → Threat Hypotheses → Risk Assessment.

## Bayesian Network Propagation Analysis

The inference traces a directed causal pathway: active Layer 1 evidence nodes (those set to E_i=1) propagated through the Noisy-OR parameterized Layer 2 behavioral indicators, elevating their activation probabilities proportional to their causal weight w_i. These elevated behavioral posteriors in turn activated Layer 3 threat hypothesis nodes (DataExfiltration, UnauthorizedAccess, ITSabotage, IPTheft), which together drove the Layer 4 RiskLevel categorical distribution through a full conditional probability table evaluated over 2^4 = 16 parent configurations.

## Noisy-OR Mathematical Interpretation

The Noisy-OR mechanism models probabilistic causation: P(Y=1|E) = 1 − (1−λ)∏(1−w_i). The leak parameter λ represents the baseline probability that Y activates even with no active parents (reflecting unmeasured causal factors). Each weight w_i captures the causal strength of parent i: higher w_i means parent i alone can cause Y with high probability. When multiple parents are simultaneously active, the product term collapses, producing a near-certain activation.

## Variable Elimination Inference

The Variable Elimination algorithm reduced the 35-node DAG to the active ancestor subgraph of RiskLevel, then eliminated hidden variables in min-degree order to minimize the induced width (treewidth bound on factor size). Each elimination step multiplied all factors containing the eliminated variable, then marginalized it out. The final unnormalized potential was normalized to obtain a valid posterior: P(RiskLevel=r|E) = Φ(r) / ΣΦ(r').

## Decision Network — Recommended Action

The decision network chose the action maximizing E[U(a)] = −C(a) − Σ_t P(T_t|E)·L_t·r_t(a), where C(a) is the cost of the action (user friction / analyst time), L_t the damage of threat t and r_t(a) the fraction of that damage left after action a. P(T_t|E) are the exact variable-elimination posteriors of the four threat nodes. The cost, loss and residual numbers are stated assumptions listed in cert_decision_network.py.

## Uncertainty & Calibration Notice

This is a probabilistic assessment, not proof of malicious intent. The model computes posterior probabilities conditioned on observed evidence under the Noisy-OR parameters learned from the CERT data. A security team should treat this as a risk score to prioritize investigation, not as definitive evidence of wrongdoing.

## Recommended Security Actions

1. **Review flagged evidence** — Validate whether the active evidence signals have legitimate business explanations.
2. **Cross-reference with HR** — Check if the user has upcoming role transitions, performance reviews, or access requirement changes.
3. **Implement the recommended security control** — Apply the decision network's recommended action proportionally.
4. **Monitor for escalation** — Set automated alerts if additional evidence signals activate in the next 24–48 hours.
5. **Document the assessment** — Record this probabilistic risk score for the user's behavioral baseline.

*Note: OpenRouter API key not configured — this is a structured template analysis. Set OPENROUTER_API_KEY in results/keys.env for full LLM-powered analysis.*
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================================")
    print(f"  CERT 35-Node Bayesian Network Web Application Server")
    print(f"  Listening on http://127.0.0.1:{port}")
    print(f"=======================================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
