import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import cert_preprocessing as P, cert_bn_builder as B, v3_common as C
OUT=C.RESULTS+os.sep; os.makedirs(OUT,exist_ok=True)
df=C.load_base(); y=df.y.values; o=np.load(os.path.join(C.RESULTS,"oof_bn.npy"))
# plots (out-of-fold)
fig,ax=plt.subplots(2,3,figsize=(16,9))
f,t,_=roc_curve(y,o); ax[0,0].plot(f,t); ax[0,0].plot([0,1],[0,1],'k:'); ax[0,0].set_title(f"ROC (AUC {roc_auc_score(y,o):.3f})"); ax[0,0].set_xlabel("FPR"); ax[0,0].set_ylabel("TPR")
p,r,th=precision_recall_curve(y,o); ax[0,1].plot(r,p); ax[0,1].set_title(f"Precision-Recall (AP {average_precision_score(y,o):.3f})"); ax[0,1].set_xlabel("Recall"); ax[0,1].set_ylabel("Precision")
q=pd.qcut(o,20,duplicates="drop"); g=pd.DataFrame({"p":o,"y":y}).groupby(q).mean(); ax[0,2].plot(g.p,g.y,'o-'); m=max(g.p.max(),g.y.max()); ax[0,2].plot([0,m],[0,m],'k:'); ax[0,2].set_title("Calibration (equal-mass bins)"); ax[0,2].set_xlabel("predicted P(High)"); ax[0,2].set_ylabel("observed attack-day rate")
ax[1,0].hist(o[y==0],bins=50,alpha=.6,label="normal",log=True); ax[1,0].hist(o[y==1],bins=50,alpha=.6,label="attack",log=True); ax[1,0].legend(); ax[1,0].set_title("Score distribution (log count)")
fs=2*p*r/(p+r+1e-9); ax[1,1].plot(th,fs[:-1],label="F1"); ax[1,1].plot(th,p[:-1],label="precision"); ax[1,1].plot(th,r[:-1],label="recall"); ax[1,1].legend(); ax[1,1].set_title("Threshold sweep"); ax[1,1].set_xlabel("threshold")
u=df.assign(s=o).groupby("user").agg(s=("s","max"),ins=("is_insider","max")); ax[1,2].hist(u.s[u.ins==0],bins=30,alpha=.6,label="normal users",log=True); ax[1,2].hist(u.s[u.ins==1],bins=30,alpha=.6,label="insiders"); ax[1,2].legend(); ax[1,2].set_title("Per-user max score")
plt.tight_layout(); plt.savefig(OUT+"v3_evaluation.png",dpi=130)
# final model on all data
R=P.FlagRules().fit(df); F=R.transform(df); b=B.CERTCPTBuilder(); params=b.fit(F,df)
R.summary().to_csv(OUT+"flag_rules.csv",index=False); 
L=["# Learned CPTs / priors (v3, fitted on all 1,000 users)\n","## Evidence priors P(flag=1) and rule\n","| Node | P(1) | Rule | TPR attack | FPR normal |","|---|---|---|---|---|"]
S=R.summary().set_index("node")
for n in B.EVIDENCE_NODES: L.append(f"| {n} | {params['nodes'][n]['prior'][1]:.4f} | {S.loc[n,'rule']} | {S.loc[n,'TPR_attack']} | {S.loc[n,'FPR_normal']} |")
L+=["\n## Noisy-OR nodes\n","| Node | Parents -> weights | Leak |","|---|---|---|"]
for n,s in params["nodes"].items():
    if "noisy_or" in s: L.append(f"| {n} | "+", ".join(f"{p}: {w:.2f}" for p,w in zip(s['parents'],s['noisy_or']['weights']))+f" | {s['noisy_or']['leak']:.3f} |")
L+=["\n## RiskLevel CPT\n","| "+" | ".join(B.RISK_PARENTS)+" | P(Low) | P(Med) | P(High) |","|"+"---|"*(len(B.RISK_PARENTS)+3)]
for c,d in params["nodes"][B.RISK_NODE]["cpt"].items(): L.append("| "+" | ".join(map(str,c))+f" | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} |")
open(OUT+"CPT_Tables.md","w").write("\n".join(L)); print("\n".join(L))
