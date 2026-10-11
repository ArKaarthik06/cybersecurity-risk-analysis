import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import itertools, numpy as np, pandas as pd
import cert_preprocessing as P, cert_bn_builder as B, v3_common as C
from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine
from cert_decision_network import DecisionNetworkEngine
df=C.load_base()
tr=next(P.grouped_folds(df))[0]; d=df.iloc[tr]
R=P.FlagRules().fit(d); params=B.CERTCPTBuilder().fit(R.transform(d),d)
bn=BayesianNetwork(); ve=VariableEliminationEngine(bn,bn.build_factors(params))
# 1. VE vs brute-force enumeration of the 4 threat nodes + risk given random evidence (full joint over latents)
rng=np.random.default_rng(0); worst=0
for _ in range(5):
    ev={n:int(v) for n,v in zip(B.EVIDENCE_NODES,rng.integers(0,2,21))}
    pv,_=ve.query("DataExfiltration",ev)
    fac=bn.build_factors(params); 
    from functools import reduce
    cond=[f.condition_on_evidence(ev) for f in fac]; hid=[n for n in bn.nodes if n not in ev]
    tot={0:0.0,1:0.0}
    for vals in itertools.product(*[bn.node_states.get(n,[0,1]) for n in hid]) if len(hid)<=14 else []:
        a=dict(zip(hid,vals)); p=1.0
        for f in cond:
            p*=f.table.get(tuple(a[v] for v in f.variables),0.0)
        tot[a["DataExfiltration"]]+=p
    if sum(tot.values())>0:
        worst=max(worst,abs(pv[1]-tot[1]/sum(tot.values())))
print("VE vs enumeration max abs diff:",worst)
# 2. gradient check
print("gradient check:",B.NoisyORLearner.check_gradient.__doc__ and "available")
# 3. decision engine sanity
de=DecisionNetworkEngine()
for p in (0.0,0.01,0.05,0.2,0.6):
    print(p, de.compute_expected_utility({}, {"DataExfiltration":p})["recommended_action"])
print("break-even MONITOR->MFA for DataExfiltration:",round(de.break_even("DataExfiltration"),4))
