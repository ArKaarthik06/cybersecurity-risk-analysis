"""
build_model.py - fit the v3 pipeline once and write the artefacts the web app loads.
    data/model_params.json   learned BN parameters (priors, Noisy-OR weights/leaks, RiskLevel CPT)
    data/flag_rules.json     the 21 evidence-flag rules
    data/samples.json        demo user-days (median-scoring attack day per scenario + a random normal day,
                             scored out-of-fold, i.e. by a model that never saw that user)
Usage:  python build_model.py [user_day_csv] [attack_days_csv]
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import cert_labels as L, cert_preprocessing as P, cert_bn_builder as B
from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine

here = os.path.dirname(os.path.abspath(__file__))
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data", "user_day_raw.csv.gz")
att = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data", "attack_days.csv")
raw = pd.read_csv(csv)
raw = raw.drop(columns=[c for c in ("is_insider", "threat_type", "scenario", "y") if c in raw.columns])
df = P.add_user_baselines(L.attach_labels(raw, L.load_attack_days(events_csv=att)))
y = df.y.values
print("user-days", len(df), "attack days", int(y.sum()))

def ser(params):
    out = {"nodes": {}}
    for n, s in params["nodes"].items():
        s = dict(s)
        if "cpt" in s:
            s["cpt"] = {",".join(map(str, k)): {str(a): float(b) for a, b in v.items()} for k, v in s["cpt"].items()}
        if "prior" in s:
            s["prior"] = {str(a): float(b) for a, b in s["prior"].items()}
        out["nodes"][n] = s
    return out

# out-of-fold scores for choosing honest demo samples
oof = np.zeros(len(df))
for tr, te in P.grouped_folds(df):
    R = P.FlagRules().fit(df.iloc[tr]); pr = B.CERTCPTBuilder().fit(R.transform(df.iloc[tr]), df.iloc[tr])
    bn = BayesianNetwork(); ve = VariableEliminationEngine(bn, bn.build_factors(pr))
    Fte = R.transform(df.iloc[te]); X = Fte[B.EVIDENCE_NODES].values
    U, inv = np.unique(X, axis=0, return_inverse=True)
    pH = np.array([ve.query(B.RISK_NODE, {n: int(v) for n, v in zip(B.EVIDENCE_NODES, u)})[0][2] for u in U])
    oof[te] = pH[inv.ravel()]

R = P.FlagRules().fit(df); F = R.transform(df); params = B.CERTCPTBuilder().fit(F, df)
json.dump(ser(params), open(os.path.join(here, "data", "model_params.json"), "w"), indent=1)
R.to_json(os.path.join(here, "data", "flag_rules.json"))

rng = np.random.default_rng(0); samples = []
for s in (1, 2, 3):
    idx = np.where((y == 1) & (df.scenario.values == s))[0]
    if len(idx) == 0: continue
    i = idx[np.argsort(oof[idx])[len(idx) // 2]]
    samples.append(i)
nm = np.where((y == 0) & (df.is_insider.values == 0))[0]; samples.append(int(rng.choice(nm)))
out = []
for i in samples:
    r = df.iloc[i]; ev = {n: int(F.iloc[i][n]) for n in B.EVIDENCE_NODES}
    tag = f"MALICIOUS INSIDER (scenario {int(r.scenario)})" if r.y == 1 else "NORMAL USER"
    out.append({"id": f"{r.user}_{r.day}", "user": r.user, "day": str(r.day), "is_insider": int(r.y),
                "label": f"{r.user} ({r.day}) - {tag}", "evidence": ev,
                "active_evidence": [k for k, v in ev.items() if v == 1], "oof_p_high": round(float(oof[i]), 4)})
json.dump({"samples": out}, open(os.path.join(here, "data", "samples.json"), "w"), indent=1)
print("wrote model_params.json, flag_rules.json, samples.json")
