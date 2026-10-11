import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings, time, pickle; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import cert_preprocessing as P, cert_bn_builder as B, cert_evaluation as EV, v3_common as C
from bayes_net import BayesianNetwork
from inference_ve import VariableEliminationEngine
def ve_scores(params, flags):
    bn=BayesianNetwork(); fac=bn.build_factors(params); ve=VariableEliminationEngine(bn,fac)
    X=flags[B.EVIDENCE_NODES].values; uniq,inv=np.unique(X,axis=0,return_inverse=True)
    pH=np.empty(len(uniq)); pT=np.empty((len(uniq),4))
    for i,u in enumerate(uniq):
        ev={n:int(v) for n,v in zip(B.EVIDENCE_NODES,u)}
        pH[i]=ve.query(B.RISK_NODE,ev)[0][2]
    return pH[inv.ravel()], len(uniq)
if __name__=="__main__":
    df=C.load_base(); y=df.y.values
    lam=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
    oof=np.zeros(len(df)); t=time.time()
    for fi,(tr,te) in enumerate(P.grouped_folds(df)):
        R=P.FlagRules().fit(df.iloc[tr]); Ftr,Fte=R.transform(df.iloc[tr]),R.transform(df.iloc[te])
        params=B.CERTCPTBuilder(lam_w=lam).fit(Ftr,df.iloc[tr]); oof[te],nu=ve_scores(params,Fte)
        print(f"fold {fi}: {nu} unique test patterns  ({time.time()-t:.0f}s)",flush=True)
    os.makedirs(C.RESULTS,exist_ok=True); np.save(os.path.join(C.RESULTS,"oof_bn.npy"),oof)
    c=EV.curve_metrics(y,oof); print({k:round(v,4) for k,v in c.items()})
    for thr in (0.5,0.3,0.1,0.05): 
        m=EV.day_metrics(y,oof,thr); print(f"thr {thr}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['F1']:.3f} acc={m['accuracy']:.4f} balAcc={m['balanced_acc']:.3f} FP={m['FP']} FN={m['FN']}")
    print("recall @ FPR 0.1%:",EV.recall_at_fpr(y,oof,0.001), " @0.5%:",EV.recall_at_fpr(y,oof,0.005))
    print("user-level max-score AUC:",round(EV.user_max_auc(df,oof),3))
    for k,thr in [(1,0.5),(2,0.5),(3,0.5),(2,0.3),(2,0.2)]:
        u=EV.user_level(df,oof,k_days=k,thr=thr); print(f"user-level k={k} thr={thr}: TP={u['TP']}/{u['insiders']} FP={u['FP']}/{u['users']-u['insiders']} acc={u['accuracy']:.3f} balAcc={u['balanced_acc']:.3f}")
