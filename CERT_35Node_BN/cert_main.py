"""
cert_main.py
============
Main orchestrator for the CERT Insider Threat Bayesian Network project.

Probabilistic Insider Threat Risk Assessment Using Bayesian Networks
with LLM-Based Explainability

Dataset: CERT Insider Threat Test Dataset R4.2 (synthetic data)
         Carnegie Mellon University SEI — doi:10.1184/R1/12841247.v1

Usage
-----
  python cert_main.py                    # full pipeline (preprocessing if needed)
  python cert_main.py --skip-preprocess  # use cached data/user_day_labeled.csv
  python cert_main.py --demo-only        # skip experiments, show 3 case reports
  python cert_main.py --help

System Architecture
-------------------
  CERT R4.2 zip (streaming)
      ↓
  User-day Feature Table
      ↓  + answers/insiders.csv
  Labeled DataFrame (insider / normal)
      ↓  user-level split
  Training Set → Thresholds → CPTs → BN
  Test Set     → Evidence   → VE/MCMC Inference → Evaluation
                                                ↓
                                         Evidence Ablation
                                                ↓
                                         LLM Explanation
"""

import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from cert_preprocessing  import (
    run_full_preprocessing, CERTPreprocessor,
    load_insider_labels, assign_threat_labels, user_level_split,
    EVIDENCE_NODES, generate_synthetic_evidence,
)
from cert_bn_builder     import CERTCPTBuilder, RISK_NODE, RISK_LABELS
from bayes_net           import BayesianNetwork
from inference_ve        import VariableEliminationEngine
from inference_mcmc      import MCMCEngine
from cert_reporter       import CERTReporter
from cert_baselines      import BaselineEvaluator
from cert_experiments    import (
    experiment1_structure_comparison,
    experiment2_exact_inference,
    experiment3_mcmc_convergence,
    experiment4_model_comparison,
    experiment5_probabilistic_quality,
    experiment6_explainability,
    _score_bn_on_df, _compute_classification_metrics,
)
from visualize_bn        import render_bayesian_network


ZIP_PATH    = "archive.zip"
DATA_DIR    = "data"
RESULTS_DIR = "results"


# ===========================================================================
# Factory helper
# ===========================================================================

def build_bn_and_engines(params: dict) -> tuple:
    """
    Given a CPT params dict, construct the BN + VE + MCMC engines.
    Returns (bn, factors, ve_engine, mcmc_engine).
    """
    bn      = BayesianNetwork()
    factors = bn.build_factors(params)
    assert bn.is_dag(), "BN graph is NOT acyclic — check edges!"
    ve      = VariableEliminationEngine(bn, factors)
    mcmc    = MCMCEngine(bn, factors)
    return bn, factors, ve, mcmc


def _ve_factory(params: dict) -> VariableEliminationEngine:
    """Convenience factory for experiment1 comparison."""
    bn, factors, ve, _ = build_bn_and_engines(params)
    return ve


# ===========================================================================
# Preprocessing step
# ===========================================================================

def step_preprocess(skip: bool = False, use_synthetic: bool = False) -> dict:
    """Run (or load cached) preprocessing pipeline."""
    cache_train = os.path.join(DATA_DIR, "train_evidence.csv")
    cache_test  = os.path.join(DATA_DIR, "test_evidence.csv")
    cache_thresh = os.path.join(DATA_DIR, "thresholds.json")

    if (skip or use_synthetic) and os.path.exists(cache_train) and os.path.exists(cache_test):
        print("\n[Preprocess] Loading cached evidence files...")
        train_ev = pd.read_csv(cache_train)
        test_ev  = pd.read_csv(cache_test)

        preprocessor = CERTPreprocessor()
        if os.path.exists(cache_thresh):
            preprocessor.load_thresholds(cache_thresh)

        print(f"  Train: {len(train_ev)} rows, {train_ev['is_insider'].sum()} insider days.")
        print(f"  Test : {len(test_ev)} rows, {test_ev['is_insider'].sum()} insider days.")

        return {
            "train_ev":     train_ev,
            "test_ev":      test_ev,
            "preprocessor": preprocessor,
            "insiders_df":  pd.DataFrame(),
            "train_raw":    None,
            "test_raw":     None,
        }

    global ZIP_PATH
    if not os.path.exists(ZIP_PATH) and os.path.exists("../archive.zip"):
        ZIP_PATH = "../archive.zip"

    if use_synthetic or not os.path.exists(ZIP_PATH):
        print("\n[Preprocess] Generating synthetic benchmark evidence dataset...")
        return generate_synthetic_evidence(cache_dir=DATA_DIR)

    result = run_full_preprocessing(
        ZIP_PATH, cache_dir=DATA_DIR, test_fraction=0.25, random_state=42
    )
    return result


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(args):
    print("=" * 70)
    print("  PROBABILISTIC INSIDER THREAT RISK ASSESSMENT")
    print("  Using Bayesian Networks with LLM Explainability")
    print("  Dataset: CERT Insider Threat Test Dataset R4.2 (synthetic)")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Visualise BN
    # ------------------------------------------------------------------
    print("\n[Step 1] Rendering 35-node BN visualization...")
    render_bayesian_network(os.path.join(RESULTS_DIR, "bayesian_network_cert.png"))

    # ------------------------------------------------------------------
    # Step 2: Preprocessing
    # ------------------------------------------------------------------
    print("\n[Step 2] Preprocessing...")
    prep = step_preprocess(skip=args.skip_preprocess, use_synthetic=getattr(args, 'synthetic', False))
    train_ev    = prep["train_ev"]
    test_ev     = prep["test_ev"]
    preprocessor = prep["preprocessor"]
    insiders_df  = prep["insiders_df"]
    train_raw    = prep.get("train_raw")
    test_raw     = prep.get("test_raw")

    # ------------------------------------------------------------------
    # Step 3: Learn CPTs from training data
    # ------------------------------------------------------------------
    print("\n[Step 3] Learning CPTs from training data...")
    builder      = CERTCPTBuilder()
    learned_params = builder.fit(train_ev)
    builder.print_summary()

    expert_params  = CERTCPTBuilder.expert_params()
    print(f"\n  Learned  BN: {len(learned_params['nodes'])} nodes")
    print(f"  Expert   BN: {len(expert_params['nodes'])} nodes")

    # ------------------------------------------------------------------
    # Step 4: Build primary (learned) BN
    # ------------------------------------------------------------------
    print("\n[Step 4] Building Bayesian Network...")
    bn, factors, ve_engine, mcmc_engine = build_bn_and_engines(learned_params)
    bn.summary()

    reporter = CERTReporter(ve_engine, mcmc_engine)

    # ------------------------------------------------------------------
    # Step 5: Quick demo on one high-risk user
    # ------------------------------------------------------------------
    print("\n[Step 5] Demo — selected high-risk insider user...")
    insider_test = test_ev[test_ev["is_insider"] == True]
    if len(insider_test) > 0:
        demo_row = insider_test.iloc[0]
        demo_ev  = {node: int(demo_row[node]) for node in EVIDENCE_NODES if node in demo_row}
        demo_user = demo_row.get("user", "UNKNOWN")
        demo_day  = demo_row.get("day",  "UNKNOWN")

        print(f"  User: {demo_user}  Day: {demo_day}")
        demo_report, demo_llm = reporter.generate_and_explain(
            demo_ev, user=demo_user, day=str(demo_day), use_llm=not args.no_llm
        )
        print(reporter.format_text_report(demo_report))
        if demo_llm:
            print(f"\n  ── LLM Explanation ──────────────────────────────────────")
            print(f"  {demo_llm[:800]}{'...' if len(demo_llm)>800 else ''}")
    else:
        print("  No insider users in test set for demo.")

    if args.demo_only:
        print("\n  [--demo-only] Skipping experiments.")
        return

    # ------------------------------------------------------------------
    # Step 6: Experiments
    # ------------------------------------------------------------------
    print("\n[Step 6] Running all experiments...")

    # Collect sample evidence for Exp 2 (variety of test rows)
    sample_evidences = []
    for _, row in test_ev.head(50).iterrows():
        sample_evidences.append(
            {node: int(row[node]) for node in EVIDENCE_NODES if node in row}
        )

    # --- Experiment 1: Structure Comparison ---
    exp1_results = experiment1_structure_comparison(
        train_ev, test_ev,
        ve_factory=_ve_factory,
        expert_params=expert_params,
        learned_params=learned_params,
        output_dir=RESULTS_DIR,
    )

    # --- Experiment 2: VE Elimination Strategy ---
    exp2_results = experiment2_exact_inference(
        ve_engine,
        sample_evidences,
        n_samples=min(20, len(sample_evidences)),
    )

    # --- Experiment 3: MCMC Convergence ---
    # Use the demo insider as reference evidence
    if len(insider_test) > 0:
        ref_ev = {node: int(insider_test.iloc[0][node])
                  for node in EVIDENCE_NODES if node in insider_test.iloc[0]}
    else:
        ref_ev = sample_evidences[0] if sample_evidences else {}

    exp3_results = experiment3_mcmc_convergence(
        ve_engine, mcmc_engine,
        evidence=ref_ev,
        sample_counts=[1_000, 5_000, 10_000, 50_000],
        burn_in=2_000,
        output_dir=RESULTS_DIR,
    )

    # --- Experiment 4: Model Comparison (Baselines + BN) ---
    print("\n[Baselines] Training conventional ML models...")
    evaluator = BaselineEvaluator(random_state=42)
    baseline_results = evaluator.fit_and_evaluate(train_ev, test_ev)
    evaluator.print_summary(baseline_results)

    # BN metrics on test set
    y_test_arr = test_ev["is_insider"].astype(int).values
    print("\n  Scoring BN on test set...")
    y_proba_bn = _score_bn_on_df(ve_engine, test_ev)
    bn_metrics = _compute_classification_metrics(y_test_arr, y_proba_bn, threshold=0.5)
    bn_metrics["y_proba"] = y_proba_bn
    bn_metrics["y_test"]  = y_test_arr

    exp4_results = experiment4_model_comparison(
        baseline_results, bn_metrics, output_dir=RESULTS_DIR
    )

    # --- Experiment 5: Probabilistic Quality ---
    exp5_results = experiment5_probabilistic_quality(
        bn_metrics, baseline_results, output_dir=RESULTS_DIR
    )

    # --- Experiment 6: Explainability ---
    exp6_reports = experiment6_explainability(
        reporter,
        test_ev,
        insiders_df=insiders_df,
        n_cases=3,
        output_dir=RESULTS_DIR,
        use_llm=not args.no_llm,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Results directory  : {RESULTS_DIR}/")
    print(f"  BN visualization   : {RESULTS_DIR}/bayesian_network_cert.png")
    print(f"  Exp3 plot          : {RESULTS_DIR}/exp3_mcmc_convergence.png")
    print(f"  Exp4 plot          : {RESULTS_DIR}/exp4_model_comparison.png")
    print(f"  Exp5 plot          : {RESULTS_DIR}/exp5_calibration.png")
    print(f"  Case reports       : {RESULTS_DIR}/exp6_case_*.json")
    print()
    print("  Key Metrics (Bayesian Network):")
    print(f"    Precision : {bn_metrics.get('precision', 0):.4f}")
    print(f"    Recall    : {bn_metrics.get('recall', 0):.4f}")
    print(f"    F1        : {bn_metrics.get('f1', 0):.4f}")
    print(f"    ROC-AUC   : {bn_metrics.get('roc_auc', 0):.4f}")
    print(f"    PR-AUC    : {bn_metrics.get('pr_auc', 0):.4f}")
    print(f"    Brier     : {bn_metrics.get('brier_score', 0):.4f}")
    print("=" * 70)


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CERT R4.2 Insider Threat Bayesian Network Pipeline"
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip streaming preprocessing and use cached data/train_evidence.csv",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate and use realistic synthetic evidence benchmark dataset for fast pipeline execution.",
    )
    parser.add_argument(
        "--demo-only",
        action="store_true",
        help="Only run the demo report; skip all experiments.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM API calls (use template fallback).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
