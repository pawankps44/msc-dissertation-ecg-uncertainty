import os, glob, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_MC = 30
CKPT_DIR = os.path.expanduser("~/protoecgnet/experiments/checkpoints/cat1_proto_classifier")
OUT = os.path.expanduser("~/protoecgnet/experiments/test_results/proto_uncertainty_cn")
os.makedirs(OUT, exist_ok=True)

def best_ckpt(d):
    best, bestv = None, -1
    for f in glob.glob(os.path.join(d, "*.ckpt")):
        m = re.search(r"val_auc=([0-9]+\.[0-9]+)", os.path.basename(f))
        if m and float(m.group(1)) > bestv:
            bestv, best = float(m.group(1)), f
    return best

def fmax(p, y):
    best = 0.0
    for thr in np.linspace(0.02, 0.98, 49):
        pr = (p >= thr).astype(int)
        tp = (pr * y).sum(); fp = (pr * (1 - y)).sum(); fn = ((1 - pr) * y).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        best = max(best, 2 * prec * rec / (prec + rec + 1e-9))
    return best

def metrics_on(p, y):
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    acc = ((p >= 0.5).astype(int) == y).mean()
    return auroc, fmax(p, y), acc

def margin_from(model, loader):
    ident = model.prototype_class_identity.detach().cpu().numpy().T
    acts, probs, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _, a = model(x.to(DEVICE))
            acts.append(a.cpu().numpy())
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    acts = np.concatenate(acts); probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(float)
    C = ident.shape[0]
    support = np.zeros((acts.shape[0], C)); competition = np.zeros((acts.shape[0], C))
    for c in range(C):
        mask = ident[c] == 1
        support[:, c] = acts[:, mask].max(axis=1)
        competition[:, c] = acts[:, ~mask].max(axis=1)
    return support - competition, probs, labels

ckpt = best_ckpt(CKPT_DIR)
print("Using checkpoint:", ckpt)
lm = load_label_mappings(custom_groups=True, prototype_category=1)
num_classes = len(lm["custom"])
_, val_loader, test_loader, _ = get_dataloaders(batch_size=32, mode="1D", sampling_rate=100,
    label_set="1", work_num=4, return_sample_ids=False, custom_groups=True,
    standardize=False, remove_baseline=True)
model = ProtoECGNet1D(num_classes=num_classes, single_class_prototype_per_class=6,
    joint_prototypes_per_border=0, proto_dim=512, backbone="resnet1d18",
    prototype_activation_function="log", latent_space_type="arc",
    add_on_layers_type="linear", class_specific=True, last_layer_connection_weight=1.0,
    m=0.05, dropout=0.35, custom_groups=True, label_set="1",
    pretrained_weights=ckpt).to(DEVICE)
model.eval()

margin_val, _, _ = margin_from(model, val_loader)
margin_te, probs, labels = margin_from(model, test_loader)
absmargin_val = np.abs(margin_val)
absmargin_te = np.abs(margin_te)
unc_margin = -absmargin_te

unc_cn = np.zeros_like(absmargin_te)
for c in range(num_classes):
    ref = np.sort(absmargin_val[:, c])
    ranks = np.searchsorted(ref, absmargin_te[:, c], side="right")
    pct = ranks / max(len(ref), 1)
    unc_cn[:, c] = 1.0 - pct

def enable_mc_dropout(m):
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()
model.eval(); enable_mc_dropout(model)
mc = []
with torch.no_grad():
    for t in range(T_MC):
        p = []
        for x, y in test_loader:
            logits, _, _ = model(x.to(DEVICE))
            p.append(torch.sigmoid(logits).cpu().numpy())
        mc.append(np.concatenate(p)); print(f"  MC pass {t+1}/{T_MC}")
unc_mc = np.stack(mc).std(0)

pred = (probs > 0.5).astype(int)
error = (pred != labels).astype(int).ravel()
methods = {"margin": unc_margin.ravel(), "class_norm": unc_cn.ravel(), "mc_dropout": unc_mc.ravel()}

lines = ["Error-detection AUROC:"]
print("\nError-detection AUROC:")
for name, u in methods.items():
    val = roc_auc_score(error, u)
    print(f"  {name:12s} {val:.4f}")
    lines.append(f"  {name:12s} {val:.4f}")

p_flat, y_flat = probs.ravel(), labels.ravel()
N = len(p_flat); n_pos = int(y_flat.sum())
sel_rows, diag_rows = [], []
for frac in [0.0, 0.05, 0.10, 0.20, 0.30]:
    k = int(round((1 - frac) * N)); cov = round(100 * k / N, 1)
    row = {"abstain": int(frac*100), "coverage": cov}
    drow = {"abstain": int(frac*100), "coverage": cov, "removed": N - k, "pos_total": n_pos}
    for name, u in methods.items():
        order = np.argsort(u)
        keep, removed = order[:k], order[k:]
        au, fm, ac = metrics_on(p_flat[keep], y_flat[keep])
        row[f"AUROC_{name}"] = round(au,4); row[f"Fmax_{name}"] = round(fm,4); row[f"Acc_{name}"] = round(ac,4)
        drow[f"pos_removed_{name}"] = int(y_flat[removed].sum())
        drow[f"neg_removed_{name}"] = int((y_flat[removed] == 0).sum())
    ra, rf = [], []
    for s in range(10):
        ridx = np.random.default_rng(s).permutation(N)[:k]
        a2, f2, _ = metrics_on(p_flat[ridx], y_flat[ridx]); ra.append(a2); rf.append(f2)
    row["AUROC_rand"] = round(float(np.mean(ra)),4); row["Fmax_rand"] = round(float(np.mean(rf)),4)
    sel_rows.append(row); diag_rows.append(drow)

sel = pd.DataFrame(sel_rows); diag = pd.DataFrame(diag_rows)
sel.to_csv(os.path.join(OUT, "selective.csv"), index=False)
diag.to_csv(os.path.join(OUT, "removed_pos_neg.csv"), index=False)
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n\n")
    f.write("=== Selective prediction ===\n" + sel.to_string(index=False) + "\n\n")
    f.write("=== Positives / negatives removed ===\n" + diag.to_string(index=False) + "\n")
print("\n=== Selective prediction ===\n", sel.to_string(index=False))
print("\n=== Positives / negatives removed at each level ===\n", diag.to_string(index=False))

for metric in ["AUROC", "Fmax"]:
    plt.figure(figsize=(7, 5))
    plt.plot(sel.coverage, sel[f"{metric}_margin"], "o-", label=f"{metric} (margin)")
    plt.plot(sel.coverage, sel[f"{metric}_class_norm"], "D-", label=f"{metric} (class-norm)")
    plt.plot(sel.coverage, sel[f"{metric}_mc_dropout"], "s-", label=f"{metric} (MC dropout)")
    plt.plot(sel.coverage, sel[f"{metric}_rand"], "^--", color="gray", label=f"{metric} (random)")
    plt.gca().invert_xaxis(); plt.xlabel("Coverage %"); plt.ylabel(f"{metric} on retained")
    plt.title(f"Selective prediction - {metric}"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"selective_{metric.lower()}.png"), dpi=140); plt.close()

plt.figure(figsize=(7, 5))
for name, mk in [("margin", "o-"), ("class_norm", "D-"), ("mc_dropout", "s-")]:
    plt.plot(diag.abstain, diag[f"pos_removed_{name}"], mk, label=name)
plt.xlabel("Abstain %"); plt.ylabel("Positives removed (count)")
plt.title(f"Positives removed vs abstention (total positives = {n_pos})")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "positives_removed.png"), dpi=140); plt.close()
print("\nSaved tables + plots to:", OUT)
