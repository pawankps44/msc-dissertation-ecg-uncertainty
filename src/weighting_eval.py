"""
Full evaluation of standard vs weighted prototype models.
For each model: AUROC, Fmax, ECE, Brier (from a clean test pass) and
error-detection AUROC + selective prediction (from 30 MC-Dropout passes).
"""
import os, glob, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
T_MC=30
CKPT_ROOT=os.path.expanduser("~/protoecgnet/experiments/checkpoints")
OUT=os.path.expanduser("~/protoecgnet/experiments/test_results/weighting_eval")
os.makedirs(OUT, exist_ok=True)
MODELS={"standard":"cat1_proto_classifier","w_clst":"cat1_proto_classifier_wclst","w_clst_sep":"cat1_proto_classifier_wcs","restr_proj":"cat1_proto_classifier_lowunc","w_bce":"cat1_proto_classifier_wbce"}

def best_ckpt(job):
    d=os.path.join(CKPT_ROOT,job); best,bv=None,-1
    for f in glob.glob(os.path.join(d,"*.ckpt")):
        m=re.search(r"val_auc=([0-9]+\.[0-9]+)",os.path.basename(f))
        if m and float(m.group(1))>bv: bv,best=float(m.group(1)),f
    return best

def macro_auc(p,y):
    a=[roc_auc_score(y[:,i],p[:,i]) for i in range(y.shape[1]) if len(np.unique(y[:,i]))>1]
    return float(np.mean(a)) if a else float("nan")

def fmax(p,y):
    b=0.0
    for t in np.linspace(0.02,0.98,49):
        pr=(p>=t).astype(int); tp=(pr*y).sum(); fp=(pr*(1-y)).sum(); fn=((1-pr)*y).sum()
        prc=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9); b=max(b,2*prc*rc/(prc+rc+1e-9))
    return float(b)

def ece_brier(p,y,nb=10):
    pf=p.ravel(); yf=y.ravel(); brier=float(np.mean((pf-yf)**2))
    edges=np.linspace(0,1,nb+1); ece=0.0; N=len(pf)
    for i in range(nb):
        lo,hi=edges[i],edges[i+1]
        m=(pf>=lo)&(pf<hi) if i<nb-1 else (pf>=lo)&(pf<=hi)
        if m.sum()>0: ece+=(m.sum()/N)*abs(pf[m].mean()-yf[m].mean())
    return float(ece),brier

def metrics_on(p,y):
    au=roc_auc_score(y,p) if len(np.unique(y))>1 else float("nan")
    return au,fmax(p,y),((p>=0.5).astype(int)==y).mean()

def enable_mc(m):
    for md in m.modules():
        if isinstance(md,nn.Dropout): md.train()

lm=load_label_mappings(custom_groups=True,prototype_category=1); num_classes=len(lm["custom"])
_,_,test_loader,_=get_dataloaders(batch_size=32,mode="1D",sampling_rate=100,label_set="1",
    work_num=4,return_sample_ids=False,custom_groups=True,standardize=False,remove_baseline=True)

def load_model(job):
    ck=best_ckpt(job)
    m=ProtoECGNet1D(num_classes=num_classes,single_class_prototype_per_class=6,joint_prototypes_per_border=0,
        proto_dim=512,backbone="resnet1d18",prototype_activation_function="log",latent_space_type="arc",
        add_on_layers_type="linear",class_specific=True,last_layer_connection_weight=1.0,m=0.05,dropout=0.35,
        custom_groups=True,label_set="1",pretrained_weights=ck).to(DEVICE)
    m.eval(); return m,ck

rows=[]; sel_all={}
for name,job in MODELS.items():
    model,ck=load_model(job); print(name,"->",ck)
    probs=[]; labels=[]
    with torch.no_grad():
        for x,y in test_loader:
            probs.append(torch.sigmoid(model(x.to(DEVICE))[0]).cpu().numpy()); labels.append(y.numpy())
    probs=np.concatenate(probs); labels=np.concatenate(labels).astype(float)
    au=macro_auc(probs,labels); fm=fmax(probs,labels); ece,brier=ece_brier(probs,labels)
    model.eval(); enable_mc(model); mc=[]
    with torch.no_grad():
        for t in range(T_MC):
            p=[]
            for x,y in test_loader: p.append(torch.sigmoid(model(x.to(DEVICE))[0]).cpu().numpy())
            mc.append(np.concatenate(p)); 
    std=np.stack(mc).std(0)
    pred=(probs>0.5).astype(int); err=(pred!=labels).astype(int).ravel()
    errauc=roc_auc_score(err,std.ravel())
    pf=probs.ravel(); yf=labels.ravel(); uf=std.ravel(); N=len(pf); sel=[]
    for frac in [0.0,0.1,0.2,0.3]:
        k=int(round((1-frac)*N)); keep=np.argsort(uf)[:k]; a,f,ac=metrics_on(pf[keep],yf[keep])
        sel.append((100*k//N,round(a,4),round(f,4),round(ac,4)))
    sel_all[name]=sel
    rows.append(dict(model=name,AUROC=round(au,4),Fmax=round(fm,4),ECE=round(ece,4),Brier=round(brier,4),
                     ErrDet_AUROC=round(errauc,4),AUROC_cov70=sel[-1][1],Fmax_cov70=sel[-1][2]))
    print(f"  {name}: AUROC {au:.4f} Fmax {fm:.4f} ECE {ece:.4f} Brier {brier:.4f} ErrDet {errauc:.4f}")

df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"weighting_comparison.csv"),index=False)
print("\n"+df.to_string(index=False))
covs=[100,90,80,70]
for mi,metric in enumerate(["AUROC","Fmax"],start=1):
    plt.figure(figsize=(7,5))
    for name in MODELS:
        plt.plot(covs,[s[mi] for s in sel_all[name]],"o-",label=name)
    plt.gca().invert_xaxis(); plt.xlabel("Coverage %"); plt.ylabel(f"{metric} on retained")
    plt.title(f"Selective prediction ({metric})"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT,f"selective_{metric.lower()}_variants.png"),dpi=140); plt.close()
print("saved to",OUT)
