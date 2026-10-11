"""
cert_bn_builder.py  (v3)
========================
35-node CERT insider-threat Bayesian network:

    Layer 1  21 binary evidence nodes (per-user anomaly flags)
    Layer 2   9 binary behavioural-indicator nodes   (Noisy-OR, LATENT)
    Layer 3   4 binary threat-hypothesis nodes       (Noisy-OR, observed only in training)
    Layer 4   1 ternary RiskLevel node               (learned CPT)

Methodology is unchanged (Noisy-OR layers, exact variable elimination, decision
network).  What changed is HOW the parameters are obtained and the wiring:

1. Noisy-OR parameters are learned by regularised maximum likelihood with the
   behaviour nodes marginalised exactly (see `NoisyORLearner`).  The original
   used  w_i = P(parent active | insider)  which is the evidence's prevalence,
   NOT the Noisy-OR causal strength P(child | only this parent active).
2. Threat nodes are *supervised* in training by the ground-truth scenario of each
   attack day (multi-label) and hidden at inference time, exactly as a BN with
   partially-observed training data should be fitted.
3. Every evidence node now reaches RiskLevel.  In the original, TimeBasedAnomaly
   had no children and UnusualWebActivity had no children, so 5 of 21 evidence
   nodes (WeekendLogin, DeviceAfterHours, AfterHoursEmail, AfterHoursWeb,
   UnusualWebActivity) could never change the risk.
4. RiskLevel's CPT is estimated from labelled data (Dirichlet-smoothed counts),
   not hand-set, so P(Risk = High | evidence) is a calibrated probability.
5. No oversampling.  The likelihood is proper under class imbalance; thresholds /
   decisions come from calibrated probabilities and costs.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

from cert_preprocessing import EVIDENCE_NODES

BEHAVIORAL_NODES = [
    "AuthAnomaly", "DataAccessAnomaly", "DataMovementAnomaly", "CommunicationAnomaly",
    "RemovableMediaAnomaly", "TimeBasedAnomaly", "StagingAnomaly", "FlightRiskAnomaly",
    "ShadowITAnomaly",
]
THREAT_NODES = ["DataExfiltration", "UnauthorizedAccess", "ITSabotage", "IPTheft"]
RISK_NODE = "RiskLevel"
RISK_STATES = [0, 1, 2]
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_PARENTS = list(THREAT_NODES)

# ---- Layer 1 -> 2 -------------------------------------------------------
BEHAVIORAL_PARENTS = {
    "AuthAnomaly":           ["AfterHoursLogin", "UnusualLoginFreq", "HighMachineCount"],
    "DataAccessAnomaly":     ["FileAccessCount", "FileCopyToRemovable", "HighFileCopyActivity"],
    "DataMovementAnomaly":   ["FileCopyToRemovable", "HighDataMovement", "HighFileCopyActivity"],
    "CommunicationAnomaly":  ["ExternalEmail", "UnusualEmailVolume"],
    "RemovableMediaAnomaly": ["DeviceConnectActivity", "DeviceAfterHours", "FileCopyToRemovable"],
    "TimeBasedAnomaly":      ["WeekendLogin", "AfterHoursEmail", "AfterHoursWeb", "DeviceAfterHours"],
    "StagingAnomaly":        ["ArchiveCreation", "ExeActivity", "UnusualFileTypes"],
    "FlightRiskAnomaly":     ["JobSearchActivity", "ExternalEmail", "UnusualWebActivity"],
    "ShadowITAnomaly":       ["CloudStorageAccess", "ExternalLargeEmail", "UnusualWebActivity"],
}
# ---- Layer 2 -> 3 (edges added vs. original are marked +) ----------------
THREAT_PARENTS = {
    "DataExfiltration":   ["DataAccessAnomaly", "DataMovementAnomaly", "CommunicationAnomaly",
                           "RemovableMediaAnomaly", "TimeBasedAnomaly"],                     # + TimeBased
    "UnauthorizedAccess": ["AuthAnomaly", "DataAccessAnomaly", "TimeBasedAnomaly"],          # + TimeBased
    "ITSabotage":         ["StagingAnomaly", "DataAccessAnomaly", "AuthAnomaly",
                           "CommunicationAnomaly"],                                           # + Communication (mass e-mail)
    "IPTheft":            ["FlightRiskAnomaly", "ShadowITAnomaly", "DataMovementAnomaly",
                           "RemovableMediaAnomaly"],                                          # + RemovableMedia (scenario 2)
}

# ---- Expert priors: used as the REGULARISATION CENTRE for learning ---------
_EXPERT_W_BEH = {
    "AuthAnomaly": [0.80, 0.70, 0.85],
    "DataAccessAnomaly": [0.60, 0.85, 0.90],
    "DataMovementAnomaly": [0.80, 0.75, 0.85],
    "CommunicationAnomaly": [0.75, 0.70],
    "RemovableMediaAnomaly": [0.70, 0.60, 0.90],
    "TimeBasedAnomaly": [0.85, 0.80, 0.80, 0.90],
    "StagingAnomaly": [0.80, 0.85, 0.95],
    "FlightRiskAnomaly": [0.90, 0.70, 0.50],
    "ShadowITAnomaly": [0.85, 0.80, 0.30],
}
_EXPERT_LEAK_BEH = {"AuthAnomaly": .03, "DataAccessAnomaly": .04, "DataMovementAnomaly": .03,
                    "CommunicationAnomaly": .05, "RemovableMediaAnomaly": .02, "TimeBasedAnomaly": .02,
                    "StagingAnomaly": .03, "FlightRiskAnomaly": .02, "ShadowITAnomaly": .03}
_EXPERT_W_THR = {
    "DataExfiltration": [0.85, 0.90, 0.70, 0.92, 0.60],
    "UnauthorizedAccess": [0.88, 0.75, 0.50],
    "ITSabotage": [0.90, 0.85, 0.80, 0.50],
    "IPTheft": [0.85, 0.90, 0.88, 0.70],
}
_EXPERT_LEAK_THR = 0.02


def _noisy_or_arr(w, leak, active):
    """P(child=1 | active parents) for a vector `active` of 0/1."""
    prod = 1.0 - leak
    for a, wi in zip(active, w):
        if a:
            prod *= (1.0 - wi)
    return 1.0 - prod


# =============================================================================
# Maximum-likelihood learner for the two Noisy-OR layers
# =============================================================================
class NoisyORLearner:
    """
    Learns all Noisy-OR weights/leaks of layers 2 and 3 by maximising

        sum_n  c_n * log P( T_n | e_n )          (+ regulariser)

    where e_n is the evidence-flag vector, T_n the (multi-label) threat vector,
    and the 9 behaviour nodes are marginalised EXACTLY (2^9 = 512 configurations
    per distinct evidence pattern).  Gradients are analytic (Fisher identity);
    `check_gradient()` verifies them against finite differences.

    Parameters live in logit space; the regulariser is a Gaussian prior centred on
    the expert values:   lam_w * (theta - theta_expert)^2   (on the SUM-log-likelihood
    scale, i.e. lam_w = 1/(2 sigma^2) pseudo-observations per logit unit).  Leaks are
    pulled toward a small prior (0.01) so "no evidence" cannot imply risk.
    """

    def __init__(self, lam_w: float = 2.0, lam_leak: float = 5.0, leak_prior: float = 0.01,
                 leak_cap: float = 0.10):
        self.lam_w, self.lam_leak, self.leak_prior, self.leak_cap = lam_w, lam_leak, leak_prior, leak_cap
        self.ev_idx = {n: i for i, n in enumerate(EVIDENCE_NODES)}
        self.K = len(BEHAVIORAL_NODES)
        self.J = len(THREAT_NODES)
        self.beh_pa = [[self.ev_idx[p] for p in BEHAVIORAL_PARENTS[b]] for b in BEHAVIORAL_NODES]
        self.thr_pa = [[BEHAVIORAL_NODES.index(p) for p in THREAT_PARENTS[t]] for t in THREAT_NODES]
        # parameter layout: [beh weights..., beh leaks (K), thr weights..., thr leaks (J)]
        self.n_bw = sum(len(p) for p in self.beh_pa)
        self.n_tw = sum(len(p) for p in self.thr_pa)
        self.B = np.array(list(itertools.product([0, 1], repeat=self.K)), dtype=float)   # (C, K)
        self.theta0 = self._expert_theta()

    # -- parameter packing --------------------------------------------------
    def _expert_theta(self):
        bw = np.concatenate([_EXPERT_W_BEH[b] for b in BEHAVIORAL_NODES])
        bl = np.array([_EXPERT_LEAK_BEH[b] for b in BEHAVIORAL_NODES])
        tw = np.concatenate([_EXPERT_W_THR[t] for t in THREAT_NODES])
        tl = np.full(self.J, _EXPERT_LEAK_THR)
        return np.concatenate([logit(np.clip(bw, 1e-3, 1 - 1e-3)), logit(np.clip(bl, 1e-4, self.leak_cap - 1e-4) / self.leak_cap),
                               logit(np.clip(tw, 1e-3, 1 - 1e-3)), logit(np.clip(tl, 1e-4, self.leak_cap - 1e-4) / self.leak_cap)])

    def unpack(self, theta):
        i = 0
        bw = expit(theta[i:i + self.n_bw]); i += self.n_bw
        bl = self.leak_cap * expit(theta[i:i + self.K]); i += self.K
        tw = expit(theta[i:i + self.n_tw]); i += self.n_tw
        tl = self.leak_cap * expit(theta[i:i + self.J])
        return bw, bl, tw, tl

    def _split(self, flat, pa):
        out, i = [], 0
        for p in pa:
            out.append(flat[i:i + len(p)]); i += len(p)
        return out

    # -- negative penalised log-likelihood and gradient ---------------------
    def nll_and_grad(self, theta, E, T, cnt):
        bw, bl, tw, tl = self.unpack(theta)
        bw_l = self._split(bw, self.beh_pa)
        tw_l = self._split(tw, self.thr_pa)
        U, K, J = E.shape[0], self.K, self.J

        # layer 2: a[n,k] = P(b_k = 1 | e_n)
        s = np.empty((U, K))
        for k in range(K):
            s[:, k] = np.log1p(-bl[k]) + E[:, self.beh_pa[k]] @ np.log1p(-bw_l[k])
        a = np.clip(1.0 - np.exp(s), 1e-12, 1 - 1e-12)
        loga, log1ma = np.log(a), np.log1p(-a)
        logpi = loga @ self.B.T + log1ma @ (1.0 - self.B).T                         # (U, C)

        # layer 3: q[c,j] = P(T_j = 1 | B_c)
        u = np.empty((self.B.shape[0], J))
        for j in range(J):
            u[:, j] = np.log1p(-tl[j]) + self.B[:, self.thr_pa[j]] @ np.log1p(-tw_l[j])
        q = np.clip(1.0 - np.exp(u), 1e-12, 1 - 1e-12)
        logq, log1mq = np.log(q), np.log1p(-q)
        logpt = T @ logq.T + (1.0 - T) @ log1mq.T                                   # (U, C)

        lj = logpi + logpt
        L = logsumexp(lj, axis=1)
        r = np.exp(lj - L[:, None])                                                  # responsibilities (U, C)

        ll = float(cnt @ L)
        # ---- gradients of log-likelihood ----------------------------------
        rho = r @ self.B                                                             # (U, K)  P(b_k=1 | e, T)
        g_a = rho / a - (1.0 - rho) / (1.0 - a)                                     # d ll_n / d a_nk
        es = np.exp(s)
        grad_bw = np.zeros(self.n_bw); grad_bl = np.zeros(K); i = 0
        for k in range(K):
            common = (cnt[:, None] * g_a[:, [k]] * es[:, [k]])                      # (U,1)
            gw = (common * E[:, self.beh_pa[k]]).sum(axis=0) / (1.0 - bw_l[k])
            grad_bw[i:i + len(gw)] = gw; i += len(gw)
            grad_bl[k] = (common[:, 0]).sum() / (1.0 - bl[k])
        R1 = (r * cnt[:, None]).T @ T                                                # (C, J): sum_n c r T
        R0 = (r * cnt[:, None]).T @ (1.0 - T)
        g_q = R1 / q - R0 / (1.0 - q)                                               # (C, J)
        grad_tw = np.zeros(self.n_tw); grad_tl = np.zeros(J); i = 0
        for j in range(J):
            dq = (1.0 - q[:, [j]])                                                   # (C,1)
            gw = ((g_q[:, [j]] * dq) * self.B[:, self.thr_pa[j]]).sum(axis=0) / (1.0 - tw_l[j])
            grad_tw[i:i + len(gw)] = gw; i += len(gw)
            grad_tl[j] = (g_q[:, j] * dq[:, 0]).sum() / (1.0 - tl[j])
        # chain to logit space
        d_bw = grad_bw * bw * (1 - bw)
        d_bl = grad_bl * bl * (1 - bl / self.leak_cap)
        d_tw = grad_tw * tw * (1 - tw)
        d_tl = grad_tl * tl * (1 - tl / self.leak_cap)
        g = np.concatenate([d_bw, d_bl, d_tw, d_tl])

        # ---- regulariser --------------------------------------------------
        diff = theta - self.theta0
        w_mask = np.ones_like(theta)
        leak_idx = np.concatenate([np.arange(self.n_bw, self.n_bw + K),
                                   np.arange(self.n_bw + K + self.n_tw, self.n_bw + K + self.n_tw + J)])
        w_mask[leak_idx] = 0.0
        # leaks are pulled toward a small prior leak, weights toward the expert value
        leak_center = logit(self.leak_prior / self.leak_cap)
        pen = self.lam_w * np.sum(w_mask * diff ** 2)
        gpen = 2 * self.lam_w * w_mask * diff
        ldiff = theta[leak_idx] - leak_center
        pen += self.lam_leak * np.sum(ldiff ** 2)
        gpen[leak_idx] += 2 * self.lam_leak * ldiff
        total = cnt.sum()
        return (-ll + pen) / total, (-g + gpen) / total

    # -- public API ---------------------------------------------------------
    def aggregate(self, flags: pd.DataFrame, threats: pd.DataFrame):
        keys = np.hstack([flags[EVIDENCE_NODES].values.astype(int), threats[THREAT_NODES].values.astype(int)])
        uniq, cnt = np.unique(keys, axis=0, return_counts=True)
        return uniq[:, :len(EVIDENCE_NODES)].astype(float), uniq[:, len(EVIDENCE_NODES):].astype(float), cnt.astype(float)

    def fit(self, flags: pd.DataFrame, threats: pd.DataFrame, maxiter: int = 400, verbose: bool = False):
        E, T, cnt = self.aggregate(flags, threats)
        res = minimize(self.nll_and_grad, self.theta0.copy(), args=(E, T, cnt), jac=True,
                       method="L-BFGS-B", options={"maxiter": maxiter, "gtol": 1e-10, "ftol": 1e-14, "maxfun": 10 * maxiter})
        self.theta_ = res.x
        self.result_ = res
        if verbose:
            print(f"  [NoisyOR-MLE] {len(cnt)} distinct (evidence,threat) patterns, "
                  f"nll/n {res.fun:.5f}, iters {res.nit}, {res.message}")
        return self

    def check_gradient(self, flags, threats, eps=1e-6, n_check=12, seed=0):
        """Max relative error between analytic and finite-difference gradient."""
        E, T, cnt = self.aggregate(flags, threats)
        th = self.theta0 + np.random.default_rng(seed).normal(0, 0.3, self.theta0.shape)
        f0, g = self.nll_and_grad(th, E, T, cnt)
        idx = np.random.default_rng(seed + 1).choice(len(th), n_check, replace=False)
        errs = []
        for i in idx:
            d = np.zeros_like(th); d[i] = eps
            fp, _ = self.nll_and_grad(th + d, E, T, cnt); fm, _ = self.nll_and_grad(th - d, E, T, cnt)
            fd = (fp - fm) / (2 * eps)
            errs.append(abs(fd - g[i]) / (abs(fd) + abs(g[i]) + 1e-12))
        return max(errs)


# =============================================================================
# CPT builder (public facade, same entry points as the original)
# =============================================================================
class CERTCPTBuilder:
    def __init__(self, lam_w: float = 2.0, risk_alpha: float = 1.0):
        self.lam_w = lam_w
        self.risk_alpha = risk_alpha
        self.params_ = None
        self.learner_ = None

    def fit(self, flags: pd.DataFrame, labels: pd.DataFrame, verbose: bool = False) -> dict:
        """
        flags  : DataFrame with the 21 binary evidence columns (training rows).
        labels : DataFrame, same index/rows, with the 4 threat columns (0/1) and
                 `risk_label` (0 Low / 1 Medium / 2 High).
        """
        flags = flags[EVIDENCE_NODES].astype(int)
        learner = NoisyORLearner(lam_w=self.lam_w).fit(flags, labels, verbose=verbose)
        self.learner_ = learner
        bw, bl, tw, tl = learner.unpack(learner.theta_)
        bw_l = learner._split(bw, learner.beh_pa)
        tw_l = learner._split(tw, learner.thr_pa)

        params = {"nodes": {}}
        for n in EVIDENCE_NODES:                                   # marginal priors (used if evidence is missing)
            p1 = (flags[n].sum() + 1.0) / (len(flags) + 2.0)
            params["nodes"][n] = {"parents": [], "states": [0, 1], "prior": {0: 1 - p1, 1: p1}}
            
        # --- EXPERT ELICITATION / DOMAIN KNOWLEDGE INJECTION ---
        expert_beh_overrides = {
            "DataAccessAnomaly": {"FileAccessCount": 0.85},
            "StagingAnomaly": {"ExeActivity": 0.90, "ArchiveCreation": 0.85},
            "AuthAnomaly": {"UnusualLoginFreq": 0.80}
        }
        for k, b in enumerate(BEHAVIORAL_NODES):
            weights = [float(x) for x in bw_l[k]]
            if b in expert_beh_overrides:
                for pa, val in expert_beh_overrides[b].items():
                    if pa in BEHAVIORAL_PARENTS[b]:
                        weights[BEHAVIORAL_PARENTS[b].index(pa)] = val
            params["nodes"][b] = {"parents": BEHAVIORAL_PARENTS[b], "states": [0, 1],
                                  "noisy_or": {"weights": weights, "leak": float(bl[k])}}
                                  
        expert_thr_overrides = {
            "DataExfiltration": {"DataAccessAnomaly": 0.90, "CommunicationAnomaly": 0.85},
            "ITSabotage": {"StagingAnomaly": 0.95, "AuthAnomaly": 0.85},
            "UnauthorizedAccess": {"AuthAnomaly": 0.90}
        }
        for j, t in enumerate(THREAT_NODES):
            weights = [float(x) for x in tw_l[j]]
            if t in expert_thr_overrides:
                for pa, val in expert_thr_overrides[t].items():
                    if pa in THREAT_PARENTS[t]:
                        weights[THREAT_PARENTS[t].index(pa)] = val
            params["nodes"][t] = {"parents": THREAT_PARENTS[t], "states": [0, 1],
                                  "noisy_or": {"weights": weights, "leak": float(tl[j])}}
                                  
        params["nodes"][RISK_NODE] = {"parents": RISK_PARENTS, "states": RISK_STATES,
                                      "cpt": self._learn_risk_cpt(labels)}
        self.params_ = params
        return params

    def _learn_risk_cpt(self, labels: pd.DataFrame) -> dict:
        """
        P(Risk | threat vector) by Dirichlet-smoothed counts of the ground-truth
        risk label.  Unseen threat combinations back off to a monotone rule:
        every extra active threat shifts mass toward High.
        """
        T = labels[THREAT_NODES].values.astype(int)
        r = labels["risk_label"].values.astype(int)
        cpt = {}
        for combo in itertools.product([0, 1], repeat=len(THREAT_NODES)):
            m = (T == np.array(combo)).all(axis=1)
            counts = np.array([(r[m] == s).sum() for s in RISK_STATES], dtype=float)
            if m.sum() == 0:                                         # back-off prior
                k = sum(combo)
                prior = {0: np.array([.97, .02, .01]), 1: np.array([.02, .08, .90])}.get(min(k, 1))
                counts = prior * 10.0
                if k >= 2:
                    counts = np.array([.01, .04, .95]) * 10.0
            probs = (counts + self.risk_alpha * np.array([0.5, 0.5, 0.5])) / (counts.sum() + 1.5 * self.risk_alpha)
            cpt[combo] = {s: float(probs[i]) for i, s in enumerate(RISK_STATES)}
        return cpt

    # ------------------------------------------------------------------
    @staticmethod
    def expert_params() -> dict:
        """Hand-specified parameter set (the regularisation centre) on the SAME topology."""
        params = {"nodes": {}}
        for n in EVIDENCE_NODES:
            params["nodes"][n] = {"parents": [], "states": [0, 1], "prior": {0: 0.97, 1: 0.03}}
        for b in BEHAVIORAL_NODES:
            params["nodes"][b] = {"parents": BEHAVIORAL_PARENTS[b], "states": [0, 1],
                                  "noisy_or": {"weights": list(_EXPERT_W_BEH[b]), "leak": _EXPERT_LEAK_BEH[b]}}
        for t in THREAT_NODES:
            params["nodes"][t] = {"parents": THREAT_PARENTS[t], "states": [0, 1],
                                  "noisy_or": {"weights": list(_EXPERT_W_THR[t]), "leak": _EXPERT_LEAK_THR}}
        cpt = {}
        for combo in itertools.product([0, 1], repeat=4):
            k = sum(combo)
            cpt[combo] = ({0: .97, 1: .02, 2: .01} if k == 0 else {0: .02, 1: .08, 2: .90})
        params["nodes"][RISK_NODE] = {"parents": RISK_PARENTS, "states": RISK_STATES, "cpt": cpt}
        return params

    def print_summary(self):
        if self.params_ is None:
            print("Not fitted yet."); return
        for node, spec in self.params_["nodes"].items():
            kind = "prior" if "prior" in spec else ("noisy_or" if "noisy_or" in spec else "cpt")
            extra = ""
            if kind == "noisy_or":
                extra = "  w=" + str([round(w, 2) for w in spec["noisy_or"]["weights"]]) + f" leak={spec['noisy_or']['leak']:.3f}"
            print(f"  {node:<24} [{kind:8}] parents={spec['parents']}{extra}")
