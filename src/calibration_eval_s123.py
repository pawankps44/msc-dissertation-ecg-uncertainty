"""
Four-way comparison: standard vs BCE-weighted, each with and without post-hoc
temperature scaling on the MC Dropout mean probabilities (temperature fit on the
validation set). Reports AUROC, Fmax, ECE, Brier, error-detection AUROC, and
selective prediction at 95/90/80/70% coverage, identically for all four.
"""
import os, glob, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
T_MC=30
CKPT_ROOT=os.path.expanduser("~/protoecgnet/experiments/checkpoints")
OUT=os.path.expanduser("~/protoecgnet/experiments/test_results/calibration_eval_s123")
os.makedirs(OUT, exist_ok=True)
MODELS={"standard":"standard_s123_clf","bce_weighted":"bce_s123_clf"}

def best_ckpt(job):
    d=os.path.join(CKPT_ROOT,job); best,bv=None,-1
    for f in glob.glob(os.path.join(d,"*.ckpt")):
        m=re.search(r"val_auc=([0-9]+\.[0-9]+)",os.path.basename(f))
        if m and float(m.group(1))>bv: bv,best=float(m.group(1)),f
    return best

def logit(p, eps=1e-6):
    p=np.clip(p,eps,1-eps); return np.log(p/(1-p))

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
    pf=p.ravel(); yf=y.ravel(); brier=float(np.mean((pf-yf)**2))
    edges=np.linspace(0,1,nb+1); ece=0.0; N=len(pf)
    for i in range(nb):
        lo,hi=edges[i],edges[i+1]
        m=(pf>=lo)&(pf<hi) if i<nb-1 else (pf>=lo)&(pf<=hi)
        if m.sum()>0: ece+=(m.sum()/N)*abs(pf[m].mean()-yf[m].mean())
    return float(ece),brier

def fit_T(z_val,y_val):
    best=(1.0,1e18)
    for T in np.linspace(0.5,5.0,181):
        p=1.0/(1.0+np.exp(-z_val/T)); p=np.clip(p,1e-6,1-1e-6)
        nll=-(y_val*np.log(p)+(1-y_val)*np.log(1-p)).mean()
        if nll<best[1]: best=(T,nll)
    return best[0]

def enable_mc(m):
    for md in m.modules():
        if isinstance(md,nn.Dropout): md.train()

def mc_run(model,loader):
    enable_mc(model); outs=[]
    with torch.no_grad():
        for t in range(T_MC):
            p=[]
            for batch in loader:
                x=batch[0].to(DEVICE); p.append(torch.sigmoid(model(x)[0]).cpu().numpy())
            outs.append(np.concatenate(p))
    return np.stack(outs)

def labels_of(loader):
    Y=[]
    for batch in loader: Y.append(batch[1].numpy())
    return np.concatenate(Y).astype(float)

lm=load_label_mappings(custom_groups=True,prototype_category=1); num_classes=len(lm["custom"])
_, val_loader, test_loader, _ = get_dataloaders(batch_size=32,mode="1D",sampling_rate=100,label_set="1",
    work_num=4,return_sample_ids=False,custom_groups=True,standardize=False,remove_baseline=True)
# clean validation pass (the default val_loader uses a weighted sampler)
val_clean=DataLoader(val_loader.dataset,batch_size=32,shuffle=False,num_workers=4)
y_val=labels_of(val_clean); y_test=labels_of(test_loader)

def load_model(job):
    ck=best_ckpt(job)
    m=ProtoECGNet1D(num_classes=num_classes,single_class_prototype_per_class=6,joint_prototypes_per_border=0,
        proto_dim=512,backbone="resnet1d18",prototype_activation_function="log",latent_space_type="arc",
        add_on_layers_type="linear",class_specific=True,last_layer_connection_weight=1.0,m=0.05,dropout=0.35,
        custom_groups=True,label_set="1",pretrained_weights=ck).to(DEVICE); m.eval(); return m

COVS=[95,90,80,70]
def evaluate(tag, probs, std, y):
    au=macro_auc(probs,y); fm=fmax(probs,y); ece,brier=ece_brier(probs,y)
    pred=(probs>0.5).astype(int); err=(pred!=y).astype(int).ravel()
    errdet=roc_auc_score(err,std.ravel())
    pf=probs.ravel(); yf=y.ravel(); uf=std.ravel(); N=len(pf); sel={}
    for cov in COVS:
        k=int(round(cov/100*N)); keep=np.argsort(uf)[:k]
        a=roc_auc_score(yf[keep],pf[keep]) if len(np.unique(yf[keep]))>1 else float("nan")
        sel[cov]=(round(a,4),round(fmax(pf[keep],yf[keep]),4))
    return dict(model=tag,AUROC=round(au,4),Fmax=round(fm,4),ECE=round(ece,4),Brier=round(brier,4),
                ErrDet_AUROC=round(errdet,4),
                **{f"selAUROC_{c}":sel[c][0] for c in COVS},**{f"selFmax_{c}":sel[c][1] for c in COVS})

rows=[]
for name,job in MODELS.items():
    model=load_model(job); print("model",name)
    mc_val=mc_run(model,val_clean); mc_test=mc_run(model,test_loader)
    mean_val=mc_val.mean(0); mean_test=mc_test.mean(0); std_test=mc_test.std(0)
    T=fit_T(logit(mean_val),y_val); print(f"  fitted temperature T={T:.3f}")
    rows.append(evaluate(name, mean_test, std_test, y_test))                 # uncalibrated
    cal_test=1.0/(1.0+np.exp(-logit(mean_test)/T))
    rows.append(evaluate(name+"_cal (T=%.2f)"%T, cal_test, std_test, y_test)) # calibrated

df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"calibration_comparison.csv"),index=False)
pd.set_option("display.width",200); pd.set_option("display.max_columns",50)
print("\n"+df.to_string(index=False))
print("\nsaved to",OUT)
