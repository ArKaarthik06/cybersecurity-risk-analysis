import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report
from v3_common import load_base
from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine
from cert_preprocessing import EVIDENCE_NODES
from cert_bn_builder import THREAT_NODES

def main():
    import cert_preprocessing as P
    df = load_base()
    # evaluate only on attack days
    attack_days = df[df.y == 1].copy()
    
    R = P.FlagRules().fit(df)
    flags = R.transform(attack_days)
    
    with open("data/model_params.json") as f:
        params = json.load(f)
        
    out = {"nodes": {}}
    for n, spec in params["nodes"].items():
        spec = dict(spec)
        if "prior" in spec: spec["prior"] = {int(k): v for k, v in spec["prior"].items()}
        if "cpt" in spec: spec["cpt"] = {tuple(int(x) for x in k.split(",")): {int(a): b for a, b in v.items()} for k, v in spec["cpt"].items()}
        out["nodes"][n] = spec
        
    bn = BayesianNetwork()
    fac = bn.build_factors(out)
    ve = VariableEliminationEngine(bn, fac)
    
    X = flags[EVIDENCE_NODES].values
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    pT = np.empty((len(uniq), 4))
    
    print(f"Evaluating {len(uniq)} unique attack patterns...")
    for i, u in enumerate(uniq):
        ev = {n: int(v) for n, v in zip(EVIDENCE_NODES, u)}
        for j, t in enumerate(THREAT_NODES):
            pT[i, j] = ve.query(t, ev)[0][1]
            
    pT_full = pT[inv]
    
    # get ground truth labels
    from cert_labels import SCENARIO_THREATS
    Y_true = np.zeros((len(attack_days), 4), dtype=int)
    for i, (_, row) in enumerate(attack_days.iterrows()):
        s = row['scenario']
        if s in SCENARIO_THREATS:
            for t in SCENARIO_THREATS[s]:
                Y_true[i, THREAT_NODES.index(t)] = 1
    
    print("--- Threat Prediction Metrics on Attack Days ---")
    for thr in [0.1, 0.15, 0.2]:
        Y_pred = (pT_full >= thr).astype(int)
        print(f"\nThreshold: {thr}")
        print(classification_report(Y_true, Y_pred, target_names=THREAT_NODES, zero_division=0))
        
if __name__ == "__main__":
    main()
