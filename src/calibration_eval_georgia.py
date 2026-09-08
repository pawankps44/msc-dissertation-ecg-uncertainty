import os, glob, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from ecg_utils_georgia import get_georgia_dataloaders, GEORGIA_CLASSES
from proto_models1D import ProtoECGNet1D
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
T_MC=30
CKPT_ROOT=os.path.expanduser("~/protoecgnet/experiments/checkpoints")
OUT=os.path.expanduser("~/protoecgnet/experiments/test_results/calibration_eval_georgia"); os.makedirs(OUT,exist_ok=True)
MODELS={"standard":"georgia_standard_clf","bce_weighted":"georgia_bce_clf"}
num_classes=len(GEORGIA_CLASSES)
def best_ckpt(job):
    d=os.path.join(CKPT_ROOT,job); best,bv=None,-1
    for f in glob.glob(os.path.join(d,"*.ckpt")):
        m=re.search(r"val_auc=([0-9]+\.[0-9]+)",os.path.basename(f))
        if m and float(m.group(1))>bv: bv,best=float(m.group(1)),f
    return best
def logit(p,e=1e-6): p=np.clip(p,e,1-e); return np.log(p/(1-p))
def macro_auc(p,y):
    a=[roc_auc_score(y[:,i],p[:,i]) for i in range(y.shape[1]) if len(np.unique(y[:,i]))>1]
    return float(np.mean(a)) if a else float("nan")
def fmax(p,y):
    b=0.0
    for t in np.linspace(0.02,0.98,49):
        pr=(p>=t).astype(int); tp=(pr*y).sum(); fp=(pr*(1-y)).sum(); fn=((1-pr)*y).sum()
        pc=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9); b=max(b,2*pc*rc/(pc+rc+1e-9))
    return float(b)
def ece_brier(p,y,nb=10):
    pf=p.ravel(); yf=y.ravel(); br=float(np.mean((pf-yf)**2)); e=0.0; N=len(pf); ed=np.linspace(0,1,nb+1)
    for i in range(nb):
        lo,hi=ed[i],ed[i+1]; m=(pf>=lo)&(pf<hi) if i<nb-1 else (pf>=lo)&(pf<=hi)
        if m.sum()>0: e+=(m.sum()/N)*abs(pf[m].mean()-yf[m].mean())
    return float(e),br
def fit_T(z,y):
    best=(1.0,1e18)
    for T in np.linspace(0.5,5.0,181):
        p=np.clip(1/(1+np.exp(-z/T)),1e-6,1-1e-6); nll=-(y*np.log(p)+(1-y)*np.log(1-p)).mean()
        if nll<best[1]: best=(T,nll)
    return best[0]
def enable_mc(m):
    for md in m.modules():
        if isinstance(md,nn.Dropout): md.train()
def mc_run(model,loader):
    enable_mc(model); outs=[]
    with torch.no_grad():
        for t in range(T_MC):
            p=[torch.sigmoid(model(b[0].to(DEVICE))[0]).cpu().numpy() for b in loader]
            outs.append(np.concatenate(p))
    return np.stack(outs)
def labels_of(loader): return np.concatenate([b[1].numpy() for b in loader]).astype(float)
_, val_loader, test_loader, _ = get_georgia_dataloaders(batch_size=32, work_num=4, return_sample_ids=False)
y_val=labels_of(val_loader); y_test=labels_of(test_loader)
def load_model(job):
    ck=best_ckpt(job)
    m=ProtoECGNet1D(num_classes=num_classes,single_class_prototype_per_class=6,joint_prototypes_per_border=0,
        proto_dim=512,backbone="resnet1d18",prototype_activation_function="log",latent_space_type="arc",
        add_on_layers_type="linear",class_specific=True,last_layer_connection_weight=1.0,m=0.05,dropout=0.35,
        custom_groups=True,label_set="georgia",pretrained_weights=ck).to(DEVICE); m.eval(); return m
COVS=[95,90,80,70]
def ev(tag,probs,std,y):
    au=macro_auc(probs,y); fm=fmax(probs,y); ece,br=ece_brier(probs,y)
    err=((probs>0.5).astype(int)!=y).astype(int).ravel(); ed=roc_auc_score(err,std.ravel())
    pf=probs.ravel(); yf=y.ravel(); uf=std.ravel(); N=len(pf); sel={}
    for c in COVS:
        k=int(round(c/100*N)); keep=np.argsort(uf)[:k]
        a=roc_auc_score(yf[keep],pf[keep]) if len(np.unique(yf[keep]))>1 else float("nan")
        sel[c]=(round(a,4),round(fmax(pf[keep],yf[keep]),4))
    return dict(model=tag,AUROC=round(au,4),Fmax=round(fm,4),ECE=round(ece,4),Brier=round(br,4),
                ErrDet=round(ed,4),**{f"sAUROC_{c}":sel[c][0] for c in COVS},**{f"sFmax_{c}":sel[c][1] for c in COVS})
rows=[]
for name,job in MODELS.items():
    model=load_model(job); print("model",name)
    mv=mc_run(model,val_loader); mt=mc_run(model,test_loader)
    mean_v=mv.mean(0); mean_t=mt.mean(0); std_t=mt.std(0)
    T=fit_T(logit(mean_v),y_val); print("  T=%.3f"%T)
    rows.append(ev(name,mean_t,std_t,y_test))
    cal=1/(1+np.exp(-logit(mean_t)/T))
    rows.append(ev(name+"_cal(T=%.2f)"%T,cal,std_t,y_test))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"georgia_calibration.csv"),index=False)
pd.set_option("display.width",220); pd.set_option("display.max_columns",50)
print("\n"+df.to_string(index=False)); print("\nsaved to",OUT)
