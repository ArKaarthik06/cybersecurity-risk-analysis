"""
cert_preprocessing.py  (v3)
===========================
Turn the user-day raw count table into the 21 binary BN evidence nodes.

What changed vs. the original and why
-------------------------------------
1. Per-user *causal, robust* baselines (CERT's own documentation: "deviations
   from a user's normal usage can be considered significant").  A flag fires when
   today's count is high relative to THAT USER's own history (rolling median/IQR of
   strictly past days, weekdays and weekends separately).  Robust statistics matter:
   an expanding mean/std absorbs a multi-week attack and hides its later days.
   Global percentile thresholds flagged heavy users every day and missed light
   users' real attacks.
2. Flag rules are fitted on TRAINING USERS only, using the training labels, by
   a transparent, information-theoretic rule: choose the candidate that
   maximises  KL( Bern(TPR) || Bern(FPR) )  (smoothed), i.e. the expected
   log-likelihood-ratio the flag contributes on attack days, subject to a hard
   cap on the normal-day firing rate and a minimum number of attack-day hits.
   (Two failure modes were measured and rejected: "max TPR under a 2 % FPR cap"
   picks needlessly loose thresholds - web activity went from FPR 1e-4 to 2e-2 for
   9 % more recall; and "max TP-FP per node" switches off weak-but-useful
   evidence that only matters in combination.)
3. User-grouped, insider-stratified cross-validation (the original had 70
   insiders and a single 25 % split => ~17 test insiders).
4. The three file nodes were near-duplicates.  The CERT readme states that EVERY
   file.csv row is "a file copy to a removable media device", so the original
   FileAccessCount, FileCopyToRemovable and HighFileCopyActivity were three
   views of one stream (the extension filter only subsets it).  They now carry
   distinct information:
       FileAccessCount       total files copied to removable media vs own baseline
       FileCopyToRemovable   sensitive-document copies (office/pdf/archive/exe)
       HighFileCopyActivity  copying while a removable device was connected
                             OUT OF HOURS (derived conjunction)
5. A "rarely-used-before" rule (`rare`) captures scenario 1: "a user who did not
   previously use removable drives or work after hours begins ...".
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

EVIDENCE_NODES = [
    "AfterHoursLogin", "UnusualLoginFreq", "WeekendLogin", "HighMachineCount",
    "DeviceConnectActivity", "DeviceAfterHours", "FileAccessCount",
    "FileCopyToRemovable", "ArchiveCreation", "ExeActivity", "UnusualFileTypes",
    "ExternalEmail", "UnusualEmailVolume", "AfterHoursEmail", "ExternalLargeEmail",
    "UnusualWebActivity", "AfterHoursWeb", "CloudStorageAccess", "JobSearchActivity",
    "HighDataMovement", "HighFileCopyActivity",
]

NODE_FEATURE = {
    "AfterHoursLogin": "after_hours_logon_count",
    "UnusualLoginFreq": "logon_count",
    "WeekendLogin": "weekend_logon_count",
    "HighMachineCount": "machine_count",
    "DeviceConnectActivity": "device_connect_count",
    "DeviceAfterHours": "after_hours_device_count",
    "FileAccessCount": "file_activity_count",
    "FileCopyToRemovable": "file_copy_to_removable_count",
    "ArchiveCreation": "archive_activity_count",
    "ExeActivity": "exe_activity_count",
    "UnusualFileTypes": "unusual_file_activity_count",
    "ExternalEmail": "external_email_count",
    "UnusualEmailVolume": "email_count",
    "AfterHoursEmail": "after_hours_email_count",
    "ExternalLargeEmail": "external_large_email_count",
    "UnusualWebActivity": "web_activity_count",
    "AfterHoursWeb": "after_hours_web_count",
    "CloudStorageAccess": "cloud_storage_count",
    "JobSearchActivity": "job_search_count",
    "HighDataMovement": "email_size_total",
    "HighFileCopyActivity": "after_hours_file_copy",   # derived: files copied on an out-of-hours device day
}
BASE_FEATURES = sorted(set(NODE_FEATURE.values()))
Z_STD_FLOOR = 0.5          # std floor (counts): a first use of a never-used device => z = 2
BASELINE_WINDOW = 180      # past same-day-type days used for the robust baseline
MIN_HISTORY = 14           # days of own history required before a baseline is trusted
Z_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)
Q_GRID = (0.90, 0.95, 0.98, 0.99, 0.995)
ZQ_CAPS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3)
CTX_Z = 2.0                # a day is "off-profile" for a feature when its robust z exceeds this
CTX_WINDOWS = (14, 30)
CTX_K = (1, 2, 3, 5)
RARE_GRID = (0.02, 0.05, 0.10)   # "used on <= x of this user's past days (same day-type)"


# ---------------------------------------------------------------------------
# Per-user causal baselines
# ---------------------------------------------------------------------------
def add_user_baselines(df: pd.DataFrame, features=None, window: int = BASELINE_WINDOW) -> pd.DataFrame:
    """
    Add per-user, causal, ROBUST baselines and derive:

        <feat>__z     (x - median_past) / (IQR_past/1.349 + floor)
        <feat>__rate  fraction of past same-day-type days on which the feature was non-zero

    * "past" = the previous `window` days of the SAME user and SAME day-type
      (weekday / weekend); the current day is never included (no look-ahead).
    * Median / IQR instead of mean / std: an insider's multi-week campaign
      (scenario 2: ~28 attack days) inflated an expanding mean+std until later
      attack days looked normal (web z fell 8.8 -> 5.6 -> 4.0 -> 3.4 -> 2.5 -> 0
      across the campaign and 30 % of scenario-2 attack days became invisible).
      Robust statistics are unaffected until attack days exceed ~50 % of the window.
    * Rows with < MIN_HISTORY prior days get z = 0 and rate = 1 (cold start).
    """
    features = features or BASE_FEATURES
    out = df.sort_values(["user", "day"]).reset_index(drop=True).copy()
    out["after_hours_file_copy"] = np.where(out["after_hours_device_count"] > 0,
                                            out["file_activity_count"], 0.0)
    out["_daytype"] = (pd.to_datetime(out["day"]).dt.dayofweek >= 5).astype(int)
    grp = [out["user"], out["_daytype"]]
    cols = {}
    for f in features:
        x = out[f].astype(float)
        past = x.groupby(grp, sort=False).shift(1)                      # strictly past, same user/day-type
        pg = past.groupby(grp, sort=False)
        roll = lambda q: pg.rolling(window, min_periods=MIN_HISTORY).quantile(q).reset_index(level=[0, 1], drop=True)
        q25, q50, q75 = roll(0.25), roll(0.50), roll(0.75)
        sigma = (q75 - q25) / 1.349
        z = (x - q50) / (sigma + Z_STD_FLOOR)
        cols[f + "__z"] = z.fillna(0.0).clip(-20, 50)
        act = (past > 0).astype(float).where(past.notna())
        rate = act.groupby(grp, sort=False).rolling(window, min_periods=MIN_HISTORY).mean().reset_index(level=[0, 1], drop=True)
        cols[f + "__rate"] = rate.fillna(1.0)
    # trailing campaign context: number of OFF-PROFILE days (z > CTX_Z) of this user in the previous
    # 14 / 30 calendar rows.  Strictly past (shift 1) => causal.
    for f in features:
        off = (cols[f + "__z"] > CTX_Z).astype(float)
        og = off.groupby(out["user"], sort=False).shift(1)
        for w in CTX_WINDOWS:
            cols[f"{f}__c{w}"] = (og.groupby(out["user"], sort=False).rolling(w, min_periods=1).sum()
                                  .reset_index(level=0, drop=True).fillna(0.0))
    out = pd.concat([out, pd.DataFrame(cols, index=out.index)], axis=1)
    return out.drop(columns=["_daytype"])


# ---------------------------------------------------------------------------
# Flag rules (fit on training users only)
# ---------------------------------------------------------------------------
class FlagRules:
    """
    Learns, for every evidence node, a binary rule from {per-user z above a
    normal-day quantile, raw above a normal-day quantile, "rarely used before",
    raw > 0}.  Selection: maximise the smoothed KL divergence between the flag's
    attack-day and normal-day firing rates (min `min_hits` attack-day hits,
    normal-day firing rate <= fpr_cap).
    """

    def __init__(self, fpr_cap: float = 0.05, min_hits: int = 5):
        self.fpr_cap = fpr_cap
        self.min_hits = min_hits
        self.rules = {}

    # candidate evaluation ---------------------------------------------------
    @staticmethod
    def _candidates(df, feat, normal_mask):
        out = []
        # z-threshold set DIRECTLY from the normal-day distribution of the per-user z-score:
        # flag iff z > (1 - cap)-quantile of normal training days  =>  FPR <= cap by construction,
        # with no coarse grid that could skip the useful operating point.
        zn = df.loc[normal_mask, feat + "__z"]
        for cap in ZQ_CAPS:
            thr = float(zn.quantile(1.0 - cap))
            out.append((("zq", cap, thr), (df[feat + "__z"] > thr)))
        for q in Q_GRID:
            thr = float(df.loc[normal_mask, feat].quantile(q))
            out.append((("q", q, thr), (df[feat] > thr)))
        for r in RARE_GRID:
            out.append((("rare", r, None), (df[feat] > 0) & (df[feat + "__rate"] <= r)))
        out.append((("any", 0, 0.0), (df[feat] > 0)))
        return out

    @staticmethod
    def _ctx_candidates(df, feat):
        # campaign-context rules: >= k off-profile days in the trailing w days AND some activity today
        return [(("ctx", k, w), (df[f"{feat}__c{w}"] >= k) & (df[feat + "__z"] > 0.5))
                for w in CTX_WINDOWS for k in CTX_K]

    @staticmethod
    def _kl(tp, n_pos, fp, n_neg):
        a = (tp + 0.5) / (n_pos + 1.0)          # smoothed P(flag | attack day)
        b = (fp + 0.5) / (n_neg + 1.0)          # smoothed P(flag | normal day)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    def fit(self, train: pd.DataFrame):
        d = train
        normal = (d["y"] == 0)
        pos = (d["y"] == 1)
        n_pos, n_neg = int(pos.sum()), int(normal.sum())
        for node in EVIDENCE_NODES:
            feat = NODE_FEATURE[node]
            best = None
            for rule, flag in self._candidates(d, feat, normal):
                tp, fp = int(flag[pos].sum()), int(flag[normal].sum())
                fpr, tpr = fp / n_neg, tp / max(n_pos, 1)
                if fpr > self.fpr_cap or tp < self.min_hits or tpr <= fpr:
                    continue
                gain = self._kl(tp, n_pos, fp, n_neg)
                if best is None or gain > best["gain"]:
                    best = {"feat": feat, "rule": rule, "tpr": tpr, "fpr": fpr, "gain": float(gain)}
            # same-day rule OR a campaign-context rule: keeps first-day detection AND sustained-campaign detection
            if best is not None:
                base_flag = [fl for rl, fl in self._candidates(d, feat, normal) if rl == best["rule"]][0]
                base_rule, base_gain = best["rule"], best["gain"]
                for rule, cf in self._ctx_candidates(d, feat):
                    flag = base_flag | cf
                    tp, fp = int(flag[pos].sum()), int(flag[normal].sum())
                    fpr, tpr = fp / n_neg, tp / max(n_pos, 1)
                    if fpr > self.fpr_cap or tpr <= fpr:
                        continue
                    gain = self._kl(tp, n_pos, fp, n_neg)
                    if gain > best["gain"] * 1.05:       # require a real (5 %) gain over the best so far
                        best = {"feat": feat, "rule": ("or", base_rule, rule), "tpr": tpr, "fpr": fpr, "gain": float(gain)}
            if best is None:                             # uninformative node -> ultra-conservative rule
                thr = float(d.loc[normal, feat].quantile(0.9995))
                flag = d[feat] > thr
                best = {"feat": feat, "rule": ("q", 0.9995, thr), "tpr": float(flag[pos].mean()),
                        "fpr": float(flag[normal].mean()), "gain": 0.0}
            self.rules[node] = best
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df
        out = {}
        def one(rule, feat):
            kind, k, thr = rule
            if kind == "zq":
                return d[feat + "__z"] > thr
            if kind == "q":
                return d[feat] > thr
            if kind == "rare":
                return (d[feat] > 0) & (d[feat + "__rate"] <= k)
            if kind == "ctx":
                return (d[f"{feat}__c{int(thr)}"] >= k) & (d[feat + "__z"] > 0.5)
            return d[feat] > 0
        for node, r in self.rules.items():
            feat = r["feat"]
            if r["rule"][0] == "or":
                out[node] = (one(r["rule"][1], feat) | one(r["rule"][2], feat)).astype(int).values
                continue
            kind, k, thr = r["rule"]
            if kind == "zq":
                f = d[feat + "__z"] > thr
            elif kind == "q":
                f = d[feat] > thr
            elif kind == "rare":
                f = (d[feat] > 0) & (d[feat + "__rate"] <= k)
            elif kind == "ctx":
                f = (d[f"{feat}__c{int(thr)}"] >= k) & (d[feat + "__z"] > 0.5)
            else:
                f = d[feat] > 0
            out[node] = f.astype(int).values
        return pd.DataFrame(out, index=df.index)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump({n: {**r, "rule": list(r["rule"])} for n, r in self.rules.items()}, fh, indent=2)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"node": n, "feature": r["feat"], "rule": (f"z>{r['rule'][2]:.2f}" if r["rule"][0] == "zq" else (f"{r['rule'][1][0]} OR ctx>={r['rule'][2][1]}/{r['rule'][2][2]}d" if r["rule"][0] == "or" else f"{r['rule'][0]} {r['rule'][1]}")),
                              "TPR_attack": round(r["tpr"], 3), "FPR_normal": round(r["fpr"], 4)}
                             for n, r in self.rules.items()])


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def grouped_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """
    User-grouped folds, stratified on (scenario) so each fold holds insiders of
    every scenario.  No user appears in both train and test.
    """
    strat = df["scenario"].values
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(df, strat, df["user"].values):
        yield tr, te
