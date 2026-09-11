
import os, glob, re, json
import numpy as np, torch
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D

SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
CKPT_ROOT=os.path.expanduser("~/protoecgnet/experiments/checkpoints")
WEIGHTS=os.path.expanduser("~/protoecgnet/experiments/preprocessing/train_uncertainty_weights.npz")
JOB="cat1_proto_joint"            # standard joint model (control)
OUTJOB="cat1_proto_proj_lowunc"
OUTDIR=os.path.join(CKPT_ROOT, OUTJOB); os.makedirs(OUTDIR, exist_ok=True)
KEEP_PCTL=50                       # keep samples with weight >= this percentile (the confident half)

def best_ckpt(job):
    d=os.path.join(CKPT_ROOT,job); best,bv=None,-1
    for f in glob.glob(os.path.join(d,"*.ckpt")):
        m=re.search(r"val_auc=([0-9]+\.[0-9]+)",os.path.basename(f))
        if m and float(m.group(1))>bv: bv,best=float(m.group(1)),f
    return best

lm=load_label_mappings(custom_groups=True,prototype_category=1); num_classes=len(lm["custom"])
train_loader,_,_,_=get_dataloaders(batch_size=32,mode="1D",sampling_rate=100,label_set="1",
    work_num=4,return_sample_ids=True,custom_groups=True,standardize=False,remove_baseline=True)

ck=best_ckpt(JOB); print("joint ckpt:",ck)
model=ProtoECGNet1D(num_classes=num_classes,single_class_prototype_per_class=6,joint_prototypes_per_border=0,
    proto_dim=512,backbone="resnet1d18",prototype_activation_function="log",latent_space_type="arc",
    add_on_layers_type="linear",class_specific=True,last_layer_connection_weight=1.0,m=0.05,dropout=0.35,
    custom_groups=True,label_set="1",pretrained_weights=ck).to(DEVICE); model.eval()

d=np.load(WEIGHTS); wdict={int(i):float(w) for i,w in zip(d["ecg_id"],d["weight"])}
thresh=float(np.percentile(list(wdict.values()), KEEP_PCTL))
print("keep samples with weight >=", round(thresh,3))

nP=model.prototype_vectors.shape[0]
ident=model.prototype_class_identity.detach().cpu().numpy()
best_s=np.full(nP,-np.inf); best_f=[None]*nP; best_m=[None]*nP     # low-uncertainty candidates
fb_s=np.full(nP,-np.inf);   fb_f=[None]*nP;   fb_m=[None]*nP       # fallback: any class example

def matches(lv, idxs): return any(lv[i]==1 for i in idxs)

with torch.no_grad():
    for X,y,ids in train_loader:
        feats,acts=model.push_forward(X.to(DEVICE))
        feats=feats.cpu().numpy(); acts=acts.cpu().numpy(); yb=y.numpy(); idn=np.asarray(ids).astype(int)
        for j in range(nP):
            cidx=np.where(ident[j]==1)[0]
            for i in range(acts.shape[0]):
                if matches(yb[i],cidx):
                    sc=acts[i,j]
                    if sc>fb_s[j]: fb_s[j]=sc; fb_f[j]=feats[i]; fb_m[j]={"ecg_id":int(idn[i])}
                    if wdict.get(int(idn[i]),1.0)>=thresh and sc>best_s[j]:
                        best_s[j]=sc; best_f[j]=feats[i]; best_m[j]={"ecg_id":int(idn[i])}

newvec=model.prototype_vectors.detach().cpu().numpy().copy(); nfb=0; meta={}
for j in range(nP):
    if best_f[j] is not None: newvec[j]=best_f[j]; meta[j]=best_m[j]
    elif fb_f[j] is not None: newvec[j]=fb_f[j]; meta[j]=fb_m[j]; nfb+=1
print("prototypes using fallback (no low-uncertainty candidate):", nfb, "/", nP)
model.prototype_vectors.data.copy_(torch.tensor(newvec, device=DEVICE))

out=os.path.join(OUTDIR, OUTJOB+"_projection.pth")
torch.save(model.state_dict(), out)
json.dump({str(k):v for k,v in meta.items()}, open(os.path.join(OUTDIR,OUTJOB+"_prototype_metadata.json"),"w"), indent=2)
print("saved projected weights to", out)
