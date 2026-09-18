"""
cert_app_server.py
==================
Flask Web Server for the 35-Node Probabilistic Insider Threat System.

Provides REST API endpoints for:
  - Interactive Bayesian Network Schema & Topology (/api/schema)
  - Pre-loaded User-Day Log Samples (/api/samples)
  - Real-time Log Evidence Extraction & BN Risk Analysis (/api/analyze)
  - Step-by-Step Mathematical Trace Generation (/api/math-trace)

Serves the front-end CyberGuard web dashboard from /static.
"""

import os
import sys
import json
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from flask import Flask, request, jsonify, send_from_directory

# Project imports
from cert_preprocessing import EVIDENCE_NODES, CERTPreprocessor
from cert_bn_builder import (
    CERTCPTBuilder, BEHAVIORAL_NODES, THREAT_NODES, RISK_NODE, RISK_LABELS, CERT_EXPERT_PRIORS,
    BEHAVIORAL_PARENTS, THREAT_PARENTS, RISK_PARENTS
)
from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine
from cert_decision_network import DecisionNetworkEngine
from cert_reporter import CERTReporter
from cert_math_tracer import MathTracer

app = Flask(__name__, static_folder="static", static_url_path="")


# ---------------------------------------------------------------------------
# Global Initialization of Model Components
# ---------------------------------------------------------------------------
print("[Server] Initializing Bayesian Network and Inference Engine...")
cpt_builder = CERTCPTBuilder()

# Load training data if available to fit CPT parameters
train_csv = os.path.join("data", "train_evidence.csv")
if os.path.exists(train_csv):
    try:
        train_df = pd.read_csv(train_csv)
        bn_params = cpt_builder.fit(train_df)
        print("[Server] CPT parameters learned from data/train_evidence.csv")
    except Exception as e:
        print(f"[Server] Using expert default params ({e})")
        bn_params = cpt_builder.expert_params()
else:
    bn_params = cpt_builder.expert_params()

bn = BayesianNetwork()
factors = bn.build_factors(bn_params)
ve_engine = VariableEliminationEngine(bn, factors)
decision_engine = DecisionNetworkEngine()
reporter = CERTReporter(ve_engine)
math_tracer = MathTracer(cpt_builder)
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
        nodes_info.append({
            "id": node,
            "label": node,
            "layer": 1,
            "layer_name": "Observable Evidence",
            "type": "evidence",
            "parents": [],
            "prior": CERT_EXPERT_PRIORS.get(node, 0.05)
        })
        
    # Layer 2: Behavioral Indicators
    for node in BEHAVIORAL_NODES:
        spec = bn_params["nodes"].get(node, {})
        parents = spec.get("parents") or BEHAVIORAL_PARENTS.get(node, [])
        nodes_info.append({
            "id": node,
            "label": node,
            "layer": 2,
            "layer_name": "Behavioral Indicators",
            "type": "indicator",
            "parents": parents
        })
        
    # Layer 3: Threat Hypotheses
    for node in THREAT_NODES:
        spec = bn_params["nodes"].get(node, {})
        parents = spec.get("parents") or THREAT_PARENTS.get(node, [])
        nodes_info.append({
            "id": node,
            "label": node,
            "layer": 3,
            "layer_name": "Threat Hypotheses",
            "type": "threat",
            "parents": parents
        })
        
    # Layer 4: Overall Risk
    spec = bn_params["nodes"].get(RISK_NODE, {})
    parents = spec.get("parents") or RISK_PARENTS
    nodes_info.append({
        "id": RISK_NODE,
        "label": RISK_NODE,
        "layer": 4,
        "layer_name": "Risk Assessment",
        "type": "risk",
        "parents": parents
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
    Returns sample user-days from test evidence CSV for demonstration.
    """
    test_csv = os.path.join("data", "test_evidence.csv")
    samples = []
    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        # Select 4 representative user days
        insiders = df[df["is_insider"] == 1].head(3)
        normals  = df[df["is_insider"] == 0].head(1)
        combined = pd.concat([insiders, normals])
        
        for idx, row in combined.iterrows():
            user = row.get("user", f"USER_{idx}")
            day = row.get("day", f"2026-02-01")
            is_insider = int(row.get("is_insider", 0))
            
            ev_dict = {col: int(row[col]) for col in EVIDENCE_NODES if col in row}
            active = [k for k, v in ev_dict.items() if v == 1]
            
            samples.append({
                "id": f"{user}_{day}",
                "user": user,
                "day": day,
                "is_insider": is_insider,
                "label": f"{user} ({day}) - {'MALICIOUS INSIDER' if is_insider else 'NORMAL USER'}",
                "evidence": ev_dict,
                "active_evidence": active
            })

    # Fallback default samples if test_csv not populated
    if not samples:
        samples = [
            {
                "id": "INSIDER_0010_2026-02-11",
                "user": "INSIDER_0010",
                "day": "2026-02-11",
                "is_insider": 1,
                "label": "INSIDER_0010 (2026-02-11) - Flight Risk & IP Theft",
                "evidence": {"FileAccessCount": 1, "ExternalEmail": 1, "AfterHoursEmail": 1},
                "active_evidence": ["FileAccessCount", "ExternalEmail", "AfterHoursEmail"]
            },
            {
                "id": "INSIDER_0001_2026-02-02",
                "user": "INSIDER_0001",
                "day": "2026-02-02",
                "is_insider": 1,
                "label": "INSIDER_0001 (2026-02-02) - Unauthorized Access",
                "evidence": {"AfterHoursLogin": 1, "UnusualLoginFreq": 1, "ExternalLargeEmail": 1, "UnusualWebActivity": 1},
                "active_evidence": ["AfterHoursLogin", "UnusualLoginFreq", "ExternalLargeEmail", "UnusualWebActivity"]
            },
            {
                "id": "NORMAL_0004_2026-02-05",
                "user": "USER_0004",
                "day": "2026-02-05",
                "is_insider": 0,
                "label": "USER_0004 (2026-02-05) - Normal Benign Baseline",
                "evidence": {},
                "active_evidence": []
            }
        ]
        
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
        
        # Ensure proper integer types for evidence
        clean_evidence = {k: int(v) for k, v in evidence.items() if k in EVIDENCE_NODES and int(v) == 1}

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
            decision_results=decision_results
        )

        return jsonify({
            "status": "success",
            "user": user_name,
            "day": day_str,
            "active_evidence": list(clean_evidence.keys()),
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
                "runtime_ms": round(stats.get("runtime_ms", 0.0), 3)
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
    """
    node_activations = {}
    
    # Layer 1
    for n in EVIDENCE_NODES:
        node_activations[n] = 1.0 if evidence.get(n, 0) == 1 else 0.05
        
    # Layer 2
    for n, p in behavioral.items():
        node_activations[n] = round(min(1.0, max(0.05, p * 1.5)), 3)
        
    # Layer 3
    for n, p in threats.items():
        node_activations[n] = round(min(1.0, max(0.05, p * 1.8)), 3)
        
    # Layer 4
    node_activations[RISK_NODE] = round(risk.get("High", 0.05), 3)

    return {
        "node_activations": node_activations,
        "active_evidence_count": len([k for k, v in evidence.items() if v == 1])
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================================")
    print(f"  CERT 35-Node Bayesian Network Web Application Server")
    print(f"  Listening on http://127.0.0.1:{port}")
    print(f"=======================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
