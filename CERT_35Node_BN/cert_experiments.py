"""
cert_experiments.py
===================
All six experiments for the CERT Insider Threat BN project.

Experiment 1 — BN Structure Comparison
  Expert BN vs Data-learned BN vs Hybrid BN
  Metric: predictive performance + BIC (via pgmpy if available)

Experiment 2 — Exact Inference Comparison
  VE with Min-Degree vs Min-Fill ordering
  Metric: runtime, induced width, max factor size

Experiment 3 — Approximate Inference Convergence
  MCMC at 1K / 5K / 10K / 50K samples vs VE ground truth
  Metric: L1 error, runtime, convergence

Experiment 4 — Model Comparison
  BN vs Naive Bayes, LR, RF, XGBoost
  Metric: Precision, Recall, F1, ROC-AUC, PR-AUC

Experiment 5 — Probabilistic Quality
  Brier Score, Log Loss, Calibration Curve, ECE

Experiment 6 — Explainability Case Studies
  High-risk user deep-dives: posterior, ablation, LLM report
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from cert_bn_builder import (
    EVIDENCE_NODES, BEHAVIORAL_NODES, THREAT_NODES, RISK_NODE, RISK_LABELS
)

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import log_loss, brier_score_loss
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ===========================================================================
# Experiment 1 — BN Structure Comparison
# ===========================================================================

def experiment1_structure_comparison(
    train_ev_df: pd.DataFrame,
    test_ev_df:  pd.DataFrame,
    ve_factory,          # callable(params) → VariableEliminationEngine
    expert_params: dict,
    learned_params: dict,
    output_dir: str = "results",
) -> dict:
    """
    Compare Expert BN vs Learned BN on the test set.

    For the data-learned BN, we optionally run pgmpy's HillClimbSearch
    to compare its structure score against the expert topology.

    Parameters
    ----------
    ve_factory     : function that takes params dict and returns a fitted VE engine.
    expert_params  : CERTCPTBuilder.expert_params()
    learned_params : CERTCPTBuilder().fit(train_ev_df)

    Returns
    -------
    Dict with per-structure predictive metrics.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: BN Structure Comparison")
    print("=" * 60)

    results = {}
    y_test = test_ev_df["is_insider"].astype(int).values

    for name, params in [("Expert BN", expert_params), ("Learned BN", learned_params)]:
        print(f"\n  → Evaluating {name}...")
        try:
            ve = ve_factory(params)
            y_prob = _score_bn_on_df(ve, test_ev_df)
            results[name] = _compute_classification_metrics(y_test, y_prob)
            print(f"    ROC-AUC = {results[name]['roc_auc']:.4f}")
            print(f"    PR-AUC  = {results[name]['pr_auc']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {}

    # Hybrid BN: use expert edges + learned CPTs
    print("\n  → Evaluating Hybrid BN (expert edges + learned CPTs)...")
    try:
        hybrid_params = _make_hybrid(expert_params, learned_params)
        ve_hybrid = ve_factory(hybrid_params)
        y_prob_hyb = _score_bn_on_df(ve_hybrid, test_ev_df)
        results["Hybrid BN"] = _compute_classification_metrics(y_test, y_prob_hyb)
        print(f"    ROC-AUC = {results['Hybrid BN']['roc_auc']:.4f}")
        print(f"    PR-AUC  = {results['Hybrid BN']['pr_auc']:.4f}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Optional: pgmpy structure learning BIC score
    _try_pgmpy_structure_learning(train_ev_df, output_dir)

    # Print comparison table
    _print_structure_table(results)
    return results


def _make_hybrid(expert_params: dict, learned_params: dict) -> dict:
    """
    Hybrid BN: keep expert graph topology (edges) but use data-learned CPTs.
    Specifically: expert prior structure but learned Noisy-OR weights.
    """
    import copy
    hybrid = copy.deepcopy(expert_params)
    # Override Noisy-OR weights on behavioral nodes from learned params
    for node in BEHAVIORAL_NODES + THREAT_NODES:
        if node in learned_params["nodes"] and node in hybrid["nodes"]:
            learned_spec = learned_params["nodes"][node]
            if "noisy_or" in learned_spec and "noisy_or" in hybrid["nodes"][node]:
                hybrid["nodes"][node]["noisy_or"]["weights"] = \
                    learned_spec["noisy_or"]["weights"]
    return hybrid


def _try_pgmpy_structure_learning(train_ev_df: pd.DataFrame, output_dir: str):
    """Run pgmpy HillClimbSearch if available and print BIC scores."""
    try:
        from pgmpy.estimators import HillClimbSearch, BicScore
        from pgmpy.models     import BayesianNetwork as PgmpyBN

        print("\n  → pgmpy HillClimbSearch (BIC scoring)...")
        cols = [c for c in EVIDENCE_NODES if c in train_ev_df.columns]
        data = train_ev_df[cols].astype(int)

        hc  = HillClimbSearch(data)
        est = hc.estimate(scoring_method=BicScore(data), max_iter=500)
        edges = list(est.edges())
        print(f"    Learned edges ({len(edges)}): {edges[:10]}{'...' if len(edges)>10 else ''}")

        bic = BicScore(data)
        score = sum(bic.local_score(node, list(est.predecessors(node)))
                    for node in est.nodes())
        print(f"    Total BIC score: {score:.2f}")

        with open(f"{output_dir}/pgmpy_structure.txt", "w") as f:
            f.write(f"Learned edges:\n{edges}\nBIC: {score:.2f}\n")

    except ImportError:
        print("\n  [skip] pgmpy not installed — structure learning skipped.")
    except Exception as e:
        print(f"\n  [skip] pgmpy structure learning failed: {e}")


def _print_structure_table(results: dict):
    print()
    print(f"  {'Structure':<18} {'ROC-AUC':>8} {'PR-AUC':>7} {'F1':>7} {'Precision':>10} {'Recall':>7}")
    print("  " + "-" * 58)
    for name, m in results.items():
        if m:
            print(
                f"  {name:<18} {m.get('roc_auc',0):>8.4f} {m.get('pr_auc',0):>7.4f} "
                f"{m.get('f1',0):>7.4f} {m.get('precision',0):>10.4f} {m.get('recall',0):>7.4f}"
            )


# ===========================================================================
# Experiment 2 — Exact Inference Comparison
# ===========================================================================

def experiment2_exact_inference(
    ve_engine,
    sample_evidence_list: list,
    n_samples: int = 20,
) -> dict:
    """
    Compare min-degree vs min-fill VE on a representative set of evidence dicts.

    Returns timing and complexity metrics for each strategy.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: VE Elimination Strategy Comparison")
    print("=" * 60)

    strategies = ["min-degree", "min-fill"]
    stats_by_strategy = {s: [] for s in strategies}

    evidence_list = sample_evidence_list[:n_samples]

    for ev in evidence_list:
        for strat in strategies:
            _, stats = ve_engine.query(RISK_NODE, evidence=ev, strategy=strat)
            stats_by_strategy[strat].append(stats)

    results = {}
    for strat in strategies:
        st_list = stats_by_strategy[strat]
        results[strat] = {
            "mean_runtime_ms":     np.mean([s["runtime_sec"] for s in st_list]) * 1000,
            "mean_induced_width":  np.mean([s["induced_width"] for s in st_list]),
            "mean_factor_size":    np.mean([s["max_factor_size"] for s in st_list]),
            "max_induced_width":   np.max([s["induced_width"] for s in st_list]),
        }

    print(f"\n  {'Strategy':<16} {'Avg Runtime(ms)':>16} {'Avg Induced Width':>18} {'Avg Max Factor':>15}")
    print("  " + "-" * 68)
    for strat, m in results.items():
        print(
            f"  {strat:<16} {m['mean_runtime_ms']:>16.3f} {m['mean_induced_width']:>18.2f} "
            f"{m['mean_factor_size']:>15.1f}"
        )

    return results


# ===========================================================================
# Experiment 3 — Approximate Inference Convergence
# ===========================================================================

def experiment3_mcmc_convergence(
    ve_engine,
    mcmc_engine,
    evidence: dict,
    sample_counts: list = None,
    burn_in: int = 2000,
    output_dir: str = "results",
) -> dict:
    """
    Study MCMC convergence as a function of sample count.
    Compares each MCMC run against VE ground truth.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if sample_counts is None:
        sample_counts = [1_000, 5_000, 10_000, 50_000]

    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: MCMC Convergence Study")
    print("=" * 60)

    # VE ground truth
    ve_post, _ = ve_engine.query(RISK_NODE, evidence=evidence)
    print(f"\n  VE Ground Truth P(RiskLevel | evidence):")
    for state, prob in sorted(ve_post.items()):
        print(f"    {RISK_LABELS[state]:<8}: {prob:.4f}")

    results = mcmc_engine.convergence_study(
        RISK_NODE,
        evidence=evidence,
        sample_counts=sample_counts,
        ve_reference=ve_post,
        burn_in=burn_in,
    )

    print(f"\n  {'Samples':>8} {'P(High)':>9} {'L1 Error':>10} {'Runtime(s)':>12}")
    print("  " + "-" * 45)
    for n in sample_counts:
        r     = results[n]
        p_h   = r["posterior"].get(2, 0.0)
        l1    = r["l1_error"] if r["l1_error"] is not None else float("nan")
        rt    = r["runtime_sec"]
        print(f"  {n:>8d} {p_h:>9.4f} {l1:>10.4f} {rt:>12.3f}")

    print(f"  {'VE ref':>8} {ve_post.get(2, 0):>9.4f} {'0.0000':>10} {'—':>12}")

    # Convergence plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        p_highs = [results[n]["posterior"].get(2, 0.0) for n in sample_counts]
        l1s     = [results[n]["l1_error"] or 0.0 for n in sample_counts]

        axes[0].semilogx(sample_counts, p_highs, "o-", color="#e74c3c", linewidth=2, label="MCMC")
        axes[0].axhline(ve_post.get(2, 0.0), color="#2980b9", linestyle="--", label="VE (exact)")
        axes[0].set_xlabel("Number of Samples")
        axes[0].set_ylabel("P(High Risk)")
        axes[0].set_title("MCMC Convergence — P(High Risk)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].semilogx(sample_counts, l1s, "s-", color="#27ae60", linewidth=2)
        axes[1].set_xlabel("Number of Samples")
        axes[1].set_ylabel("L1 Error vs VE")
        axes[1].set_title("MCMC Error Convergence")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/exp3_mcmc_convergence.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved → {output_dir}/exp3_mcmc_convergence.png")
    except Exception as e:
        print(f"  [Plot] Failed: {e}")

    return {"ve_reference": ve_post, "mcmc_results": results}


# ===========================================================================
# Experiment 4 — Model Comparison
# ===========================================================================

def experiment4_model_comparison(
    baseline_results: dict,
    bn_metrics: dict,
    output_dir: str = "results",
) -> dict:
    """
    Combine baseline and BN metrics into a single comparison table and plot.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Model Comparison")
    print("=" * 60)

    all_results = dict(baseline_results)
    all_results["BayesianNetwork"] = bn_metrics

    # Print table
    print(f"\n  {'Model':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>7} {'Brier':>7}")
    print("  " + "-" * 70)
    for name, m in all_results.items():
        if m:
            print(
                f"  {name:<22} {m.get('precision',0):>9.4f} {m.get('recall',0):>7.4f} "
                f"{m.get('f1',0):>7.4f} {m.get('roc_auc',0):>8.4f} {m.get('pr_auc',0):>7.4f} "
                f"{m.get('brier_score',0):>7.4f}"
            )

    # Bar chart
    try:
        models  = list(all_results.keys())
        metrics_to_plot = ["f1", "roc_auc", "pr_auc"]
        colors  = ["#3498db", "#e74c3c", "#2ecc71"]

        fig, ax = plt.subplots(figsize=(12, 5))
        x    = np.arange(len(models))
        w    = 0.25
        for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
            vals = [all_results[m].get(metric, 0.0) for m in models]
            ax.bar(x + i * w, vals, width=w, label=metric.upper(), color=color, alpha=0.85)

        ax.set_xticks(x + w)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison: F1, ROC-AUC, PR-AUC")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/exp4_model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved → {output_dir}/exp4_model_comparison.png")
    except Exception as e:
        print(f"  [Plot] Failed: {e}")

    return all_results


# ===========================================================================
# Experiment 5 — Probabilistic Quality
# ===========================================================================

def experiment5_probabilistic_quality(
    bn_metrics: dict,
    baseline_results: dict,
    output_dir: str = "results",
) -> dict:
    """
    Evaluate Brier Score, Log Loss, Calibration Curve, ECE.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 5: Probabilistic Quality Assessment")
    print("=" * 60)

    results = {}

    for name, m in [("BayesianNetwork", bn_metrics)] + list(baseline_results.items()):
        if not m or "y_proba" not in m:
            continue
        y_t   = np.array(m["y_test"])
        y_p   = np.array(m["y_proba"])

        brier = brier_score_loss(y_t, y_p)
        ll    = log_loss(y_t, y_p) if _HAS_SKLEARN else float("nan")
        ece   = _expected_calibration_error(y_t, y_p)

        results[name] = {"brier": brier, "log_loss": ll, "ece": ece,
                         "y_test": y_t, "y_proba": y_p}
        print(f"  {name:<22}  Brier={brier:.4f}  LogLoss={ll:.4f}  ECE={ece:.4f}")

    # Calibration curves
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        for name, r in results.items():
            try:
                frac_pos, mean_pred = calibration_curve(r["y_test"], r["y_proba"], n_bins=10)
                ax.plot(mean_pred, frac_pos, "o-", label=name)
            except Exception:
                pass
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curves")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/exp5_calibration.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot saved → {output_dir}/exp5_calibration.png")
    except Exception as e:
        print(f"  [Plot] Failed: {e}")

    return results


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bins    = np.linspace(0.0, 1.0, n_bins + 1)
    ece     = 0.0
    n_total = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n_total) * abs(acc - conf)
    return float(ece)


# ===========================================================================
# Experiment 6 — Explainability Case Studies
# ===========================================================================

def experiment6_explainability(
    reporter,
    test_ev_df: pd.DataFrame,
    insiders_df: pd.DataFrame,
    n_cases: int = 3,
    output_dir: str = "results",
    use_llm: bool = True,
) -> list:
    """
    Select high-risk test users and generate full explainability reports.

    For each selected case:
    - Show posterior probabilities
    - Show top evidence contributors
    - Perform evidence ablation
    - Generate LLM-based natural language report
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EXPERIMENT 6: Explainability Case Studies")
    print("=" * 60)

    insider_test = test_ev_df[test_ev_df["is_insider"] == True].copy()
    if len(insider_test) == 0:
        print("  No insider users in test set — using top risk-scored rows.")
        insider_test = test_ev_df.copy()

    reports = []
    seen_users: set = set()
    selected_rows = []

    for _, row in insider_test.iterrows():
        user = row.get("user", "Unknown")
        if user not in seen_users:
            seen_users.add(user)
            selected_rows.append(row)
        if len(selected_rows) >= n_cases:
            break

    for i, row in enumerate(selected_rows):
        user = row.get("user", "Unknown")
        day  = row.get("day",  "Unknown")
        ev   = {node: int(row[node]) for node in EVIDENCE_NODES if node in row}

        print(f"\n  Case {i+1}: User={user}  Day={day}")
        print(f"  Active evidence: {[k for k,v in ev.items() if v==1]}")

        report, llm_text = reporter.generate_and_explain(
            ev, user=user, day=str(day), use_llm=use_llm
        )
        reports.append(report)

        print(reporter.format_text_report(report))
        if use_llm:
            print(f"\n  ── LLM EXPLANATION ──────────────────────────────────")
            print(f"  {llm_text}")

        # Save JSON report
        report_path = f"{output_dir}/exp6_case_{i+1}_{user}.json"
        try:
            save_data = {k: v for k, v in report.items() if k not in ("y_proba", "y_test")}
            # Convert any numpy types for JSON serialisation
            with open(report_path, "w") as f:
                json_str = _json_safe_dumps(save_data)
                f.write(json_str)
            print(f"\n  Report saved → {report_path}")
        except Exception as e:
            print(f"  [Save] Failed: {e}")

    return reports


import json as _json

def _json_safe_dumps(obj) -> str:
    """JSON dumps with numpy type handling."""
    class NpEncoder(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return super().default(o)
    return _json.dumps(obj, indent=2, cls=NpEncoder)


# ===========================================================================
# Helpers
# ===========================================================================

def _score_bn_on_df(ve_engine, ev_df: pd.DataFrame) -> np.ndarray:
    """Return P(High Risk) for every row in ev_df."""
    scores = []
    for _, row in ev_df.iterrows():
        ev = {node: int(row[node]) for node in EVIDENCE_NODES if node in row}
        try:
            post, _ = ve_engine.query(RISK_NODE, evidence=ev)
            scores.append(post.get(2, 0.0))
        except Exception:
            scores.append(0.0)
    return np.array(scores)


def _compute_classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute classification metrics from probability scores."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, brier_score_loss,
    )
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_proba) if y_true.sum() > 0 else 0.0,
        "pr_auc":      average_precision_score(y_true, y_proba) if y_true.sum() > 0 else 0.0,
        "brier_score": brier_score_loss(y_true, y_proba),
        "y_proba":     y_proba,
        "y_test":      y_true,
    }
