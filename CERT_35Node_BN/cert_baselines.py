"""
cert_baselines.py
=================
Baseline ML model comparison for the CERT Insider Threat project.

Trains and evaluates four conventional classifiers on the same binary
evidence feature vector used by the Bayesian Network, enabling an
apples-to-apples comparison.

Models
------
  - Naive Bayes (GaussianNB)
  - Logistic Regression
  - Random Forest
  - XGBoost (if installed; else skipped gracefully)

Metrics
-------
  Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score
  (all computed with the positive class = is_insider = 1)

Usage
-----
  from cert_baselines import BaselineEvaluator
  be = BaselineEvaluator()
  results = be.fit_and_evaluate(train_ev_df, test_ev_df)
  be.print_summary(results)
"""

import numpy as np
import pandas as pd
import warnings

from sklearn.naive_bayes        import GaussianNB
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix,
)
from sklearn.calibration        import calibration_curve
from sklearn.preprocessing      import label_binarize

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    warnings.warn("xgboost not installed — XGBoost baseline will be skipped.", stacklevel=2)

from cert_bn_builder import EVIDENCE_NODES


class BaselineEvaluator:
    """
    Fits and evaluates conventional ML baselines on the CERT binary
    insider-threat classification task.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_ = {}

    # ------------------------------------------------------------------
    def _get_models(self) -> dict:
        models = {
            "NaiveBayes": GaussianNB(),
            "LogisticRegression": LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=self.random_state
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }
        if _HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=10,   # account for class imbalance
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.random_state,
                verbosity=0,
            )
        return models

    # ------------------------------------------------------------------
    def _prepare_xy(self, ev_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix X and binary label y from evidence DataFrame."""
        feature_cols = [c for c in EVIDENCE_NODES if c in ev_df.columns]
        X = ev_df[feature_cols].values.astype(float)
        y = ev_df["is_insider"].astype(int).values
        return X, y

    # ------------------------------------------------------------------
    def _evaluate_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
    ) -> dict:
        """Fit a model and compute all evaluation metrics."""
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "precision":       precision_score(y_test, y_pred, zero_division=0),
            "recall":          recall_score(y_test, y_pred, zero_division=0),
            "f1":              f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":         roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.0,
            "pr_auc":          average_precision_score(y_test, y_proba) if y_test.sum() > 0 else 0.0,
            "brier_score":     brier_score_loss(y_test, y_proba),
            "n_train_insider": int(y_train.sum()),
            "n_test_insider":  int(y_test.sum()),
            "n_train_total":   int(len(y_train)),
            "n_test_total":    int(len(y_test)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_proba":         y_proba,   # for calibration plots
            "y_test":          y_test,
        }
        return metrics

    # ------------------------------------------------------------------
    def fit_and_evaluate(
        self,
        train_ev_df: pd.DataFrame,
        test_ev_df:  pd.DataFrame,
    ) -> dict:
        """
        Fit all baseline models on training data and evaluate on test data.

        Parameters
        ----------
        train_ev_df : Training evidence DataFrame (includes EVIDENCE_NODES + is_insider).
        test_ev_df  : Test evidence DataFrame.

        Returns
        -------
        Dict {model_name: metrics_dict}
        """
        X_train, y_train = self._prepare_xy(train_ev_df)
        X_test,  y_test  = self._prepare_xy(test_ev_df)

        print(f"  Class distribution — Train: {int(y_train.sum())} insider / {len(y_train)} total"
              f"  ({100*y_train.mean():.1f}%)")
        print(f"  Class distribution — Test : {int(y_test.sum())} insider / {len(y_test)} total"
              f"  ({100*y_test.mean():.1f}%)")

        results = {}
        for name, model in self._get_models().items():
            print(f"  Training {name}...", end=" ", flush=True)
            try:
                metrics = self._evaluate_model(model, X_train, y_train, X_test, y_test)
                self.models_[name] = model
                results[name] = metrics
                print(f"F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}")
            except Exception as e:
                print(f"FAILED: {e}")

        return results

    # ------------------------------------------------------------------
    def print_summary(self, results: dict):
        """Print a formatted comparison table of all baselines."""
        header = f"{'Model':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} {'ROC-AUC':>8} {'PR-AUC':>7} {'Brier':>7}"
        print()
        print("=" * 75)
        print("  BASELINE MODEL COMPARISON — CERT Insider Threat Classification")
        print("=" * 75)
        print(header)
        print("-" * 75)
        for name, m in results.items():
            print(
                f"  {name:<20} {m['precision']:>9.4f} {m['recall']:>7.4f} "
                f"{m['f1']:>7.4f} {m['roc_auc']:>8.4f} {m['pr_auc']:>7.4f} "
                f"{m['brier_score']:>7.4f}"
            )
        print("=" * 75)
        print()

    # ------------------------------------------------------------------
    def evaluate_bn_as_baseline(
        self,
        ve_engine,
        test_ev_df: pd.DataFrame,
        risk_threshold: float = 0.5,
    ) -> dict:
        """
        Evaluate the Bayesian Network as a classifier by treating
        P(High Risk) > threshold as a positive (insider) prediction.

        Appends the BN result to the same metrics structure so it can
        be compared with the ML baselines in the same table.

        Parameters
        ----------
        ve_engine       : VariableEliminationEngine (fitted).
        test_ev_df      : Test evidence DataFrame (includes is_insider, EVIDENCE_NODES).
        risk_threshold  : P(High Risk) threshold for positive prediction.

        Returns
        -------
        Metrics dict (same structure as baseline model metrics).
        """
        from cert_bn_builder import EVIDENCE_NODES, RISK_NODE

        y_test  = test_ev_df["is_insider"].astype(int).values
        y_proba = []
        y_pred  = []

        print("  Evaluating BN on test set...", end=" ", flush=True)
        for _, row in test_ev_df.iterrows():
            ev = {node: int(row[node]) for node in EVIDENCE_NODES if node in row}
            try:
                post, _ = ve_engine.query(RISK_NODE, evidence=ev)
                p_high  = post.get(2, 0.0)
            except Exception:
                p_high = 0.0
            y_proba.append(p_high)
            y_pred.append(1 if p_high > risk_threshold else 0)

        y_proba = np.array(y_proba)
        y_pred  = np.array(y_pred)

        metrics = {
            "precision":       precision_score(y_test, y_pred, zero_division=0),
            "recall":          recall_score(y_test, y_pred, zero_division=0),
            "f1":              f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":         roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.0,
            "pr_auc":          average_precision_score(y_test, y_proba) if y_test.sum() > 0 else 0.0,
            "brier_score":     brier_score_loss(y_test, y_proba),
            "n_test_insider":  int(y_test.sum()),
            "n_test_total":    int(len(y_test)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_proba":         y_proba,
            "y_test":          y_test,
        }
        print(f"F1={metrics['f1']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}")
        return metrics
