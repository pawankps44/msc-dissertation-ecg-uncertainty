import os, glob, re
import numpy as np
import torch, torch.nn as nn
from ecg_utils_chapman import get_chapman_dataloaders, CHAPMAN_CLASSES
from proto_models1D import ProtoECGNet1D

SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
T_MC=30
CKPT_DIR=os.path.expanduser("~/protoecgnet/experiments/checkpoints/chapman_standard_clf")
OUT=os.path.expanduser("~/protoecgnet/experiments/preprocessing")

def best_ckpt(d):
    best,bv=None,-1
    for f in glob.glob(os.path.join(d,"*.ckpt")):
        m=re.search(r"val_auc=([0-9]+\.[0-9]+)",os.path.basename(f))
        if m and float(m.group(1))>bv: bv,best=float(m.group(1)),f
    return best

ckpt=best_ckpt(CKPT_DIR); print("ckpt", ckpt)
num_classes=len(CHAPMAN_CLASSES)
train_loader,_,_,_=get_chapman_dataloaders(batch_size=32, work_num=4, return_sample_ids=True)
model=ProtoECGNet1D(num_classes=num_classes, single_class_prototype_per_class=6,
    joint_prototypes_per_border=0, proto_dim=512, backbone="resnet1d18",
    prototype_activation_function="log", latent_space_type="arc", add_on_layers_type="linear",
    class_specific=True, last_layer_connection_weight=1.0, m=0.05, dropout=0.35,
    custom_groups=True, label_set="chapman", pretrained_weights=ckpt).to(DEVICE)
model.eval()

ids_all=[]
with torch.no_grad():
    for x,y,ids in train_loader:
        ids_all.append(np.asarray(ids).astype(np.int64))
uids=np.unique(np.concatenate(ids_all)); id2row={int(s):i for i,s in enumerate(uids)}; N=len(uids)
print("train ECGs", N)

def enable_mc(m):
    for md in m.modules():
        if isinstance(md,nn.Dropout): md.train()
model.eval(); enable_mc(model)
mc=np.zeros((N,T_MC,num_classes),dtype=np.float32)
with torch.no_grad():
    for t in range(T_MC):
        for x,y,ids in train_loader:
            p=torch.sigmoid(model(x.to(DEVICE))[0]).cpu().numpy(); ids=np.asarray(ids).astype(np.int64)
            for j,sid in enumerate(ids): mc[id2row[int(sid)],t]=p[j]
        print(f"  MC pass {t+1}/{T_MC}")
unc=mc.std(1).mean(1)
lo,hi=np.percentile(unc,1),np.percentile(unc,99)
un=np.clip((unc-lo)/(hi-lo+1e-9),0.0,1.0); weight=1.0-un
np.savez(os.path.join(OUT,"chapman_train_weights.npz"), ecg_id=uids, uncertainty=unc, weight=weight)
print("weight stats -> min %.3f mean %.3f max %.3f"%(weight.min(),weight.mean(),weight.max()))
print("saved", os.path.join(OUT,"chapman_train_weights.npz"))
