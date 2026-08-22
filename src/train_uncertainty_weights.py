"""
Step 1 of uncertainty-weighted training.
Runs MC Dropout over the TRAINING set with the standard prototype model,
gets one uncertainty value per training ECG (keyed by ecg_id), and turns it
into weight = 1 - normalized_uncertainty. Saves a lookup the training will use.
"""
import os, glob, re
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_MC = 30
CKPT_DIR = os.path.expanduser("~/protoecgnet/experiments/checkpoints/cat1_proto_classifier")
OUT = os.path.expanduser("~/protoecgnet/experiments/preprocessing")
os.makedirs(OUT, exist_ok=True)

def best_ckpt(d):
    best, bestv = None, -1
    for f in glob.glob(os.path.join(d, "*.ckpt")):
        m = re.search(r"val_auc=([0-9]+\.[0-9]+)", os.path.basename(f))
        if m and float(m.group(1)) > bestv:
            bestv, best = float(m.group(1)), f
    return best

ckpt = best_ckpt(CKPT_DIR)
print("Using checkpoint:", ckpt)
lm = load_label_mappings(custom_groups=True, prototype_category=1)
num_classes = len(lm["custom"])

# training loader WITH sample ids (so weights can be matched by ecg_id later)
train_loader, _, _, _ = get_dataloaders(batch_size=32, mode="1D", sampling_rate=100,
    label_set="1", work_num=4, return_sample_ids=True, custom_groups=True,
    standardize=False, remove_baseline=True)

model = ProtoECGNet1D(num_classes=num_classes, single_class_prototype_per_class=6,
    joint_prototypes_per_border=0, proto_dim=512, backbone="resnet1d18",
    prototype_activation_function="log", latent_space_type="arc",
    add_on_layers_type="linear", class_specific=True, last_layer_connection_weight=1.0,
    m=0.05, dropout=0.35, custom_groups=True, label_set="1",
    pretrained_weights=ckpt).to(DEVICE)
model.eval()

# first pass (no dropout) just to collect the set of ecg_ids and fix an order
ids_all = []
with torch.no_grad():
    for x, y, ids in train_loader:
        ids_all.append(np.asarray(ids).astype(np.int64))
uids = np.unique(np.concatenate(ids_all))
id_to_row = {int(s): i for i, s in enumerate(uids)}
N = len(uids)
print("training ECGs:", N)

# T_MC dropout-on passes; store each ECG's probability vector per pass, keyed by id
def enable_mc_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
model.eval(); enable_mc_dropout(model)
mc = np.zeros((N, T_MC, num_classes), dtype=np.float32)
with torch.no_grad():
    for t in range(T_MC):
        for x, y, ids in train_loader:
            p = torch.sigmoid(model(x.to(DEVICE))[0]).cpu().numpy()
            ids = np.asarray(ids).astype(np.int64)
            for j, sid in enumerate(ids):
                mc[id_to_row[int(sid)], t] = p[j]
        print(f"  MC pass {t+1}/{T_MC}")

# per-ECG uncertainty = average across classes of the std across passes
unc = mc.std(axis=1).mean(axis=1)          # (N,)
# robust min-max normalization (clip at 1st/99th percentile to ignore outliers)
lo, hi = np.percentile(unc, 1), np.percentile(unc, 99)
unc_norm = np.clip((unc - lo) / (hi - lo + 1e-9), 0.0, 1.0)
weight = 1.0 - unc_norm                     # confident ECG -> ~1, ambiguous ECG -> ~0

np.savez(os.path.join(OUT, "train_uncertainty_weights.npz"),
         ecg_id=uids, uncertainty=unc, weight=weight)
print("weight stats -> min %.3f  mean %.3f  max %.3f" % (weight.min(), weight.mean(), weight.max()))

plt.figure(figsize=(6,4))
plt.hist(weight, bins=40)
plt.xlabel("training-sample weight (1 - normalized uncertainty)"); plt.ylabel("count")
plt.title("Distribution of training weights"); plt.tight_layout()
plt.savefig(os.path.join(OUT, "train_weights_hist.png"), dpi=140); plt.close()
print("Saved weights to", os.path.join(OUT, "train_uncertainty_weights.npz"))
