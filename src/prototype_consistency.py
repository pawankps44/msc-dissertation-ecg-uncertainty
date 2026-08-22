"""
Prototype class-consistency (purity) for standard vs weighted models.
For each prototype (assigned to class c), take the top-K test ECGs it activates
most, and measure the fraction that actually carry class c. Mean over prototypes
= how 'clean' / class-consistent the prototype set is.
"""
import os, glob, re
import numpy as np, pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
K=20
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

lm=load_label_mappings(custom_groups=True,prototype_category=1); num_classes=len(lm["custom"])
class_names=lm["custom"]
_,_,test_loader,_=get_dataloaders(batch_size=32,mode="1D",sampling_rate=100,label_set="1",
    work_num=4,return_sample_ids=False,custom_groups=True,standardize=False,remove_baseline=True)

def acts_labels(model):
    A=[]; Y=[]
    with torch.no_grad():
        for x,y in test_loader:
            A.append(model(x.to(DEVICE))[2].cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(A), np.concatenate(Y).astype(float)

rows=[]; perclass={}
for name,job in MODELS.items():
    ck=best_ckpt(job); print(name,"->",ck)
    model=ProtoECGNet1D(num_classes=num_classes,single_class_prototype_per_class=6,joint_prototypes_per_border=0,
        proto_dim=512,backbone="resnet1d18",prototype_activation_function="log",latent_space_type="arc",
        add_on_layers_type="linear",class_specific=True,last_layer_connection_weight=1.0,m=0.05,dropout=0.35,
        custom_groups=True,label_set="1",pretrained_weights=ck).to(DEVICE); model.eval()
    ident=model.prototype_class_identity.detach().cpu().numpy(); proto_class=ident.argmax(1)
    acts,labels=acts_labels(model)
    pur=np.array([labels[np.argsort(-acts[:,p])[:K], proto_class[p]].mean() for p in range(len(proto_class))])
    # per-class average purity
    pc=np.array([pur[proto_class==c].mean() if (proto_class==c).any() else np.nan for c in range(num_classes)])
    perclass[name]=pc
    rows.append(dict(model=name, prototype_purity=round(float(pur.mean()),4), min_purity=round(float(pur.min()),4)))
    print(f"  {name}: mean prototype purity {pur.mean():.4f}")

df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT,"prototype_consistency.csv"),index=False)
print("\n"+df.to_string(index=False))

# bar chart of overall purity
plt.figure(figsize=(6,4))
plt.bar(list(MODELS.keys()), [r["prototype_purity"] for r in rows])
plt.ylabel("Mean prototype purity (top-%d)" % K); plt.title("Prototype class-consistency")
plt.ylim(0,1); plt.tight_layout(); plt.savefig(os.path.join(OUT,"prototype_purity.png"),dpi=140); plt.close()

# per-class purity table
pcdf=pd.DataFrame(perclass, index=class_names).round(3)
pcdf.to_csv(os.path.join(OUT,"prototype_consistency_perclass.csv"))
print("\nper-class purity:\n"+pcdf.to_string())
print("saved to",OUT)
