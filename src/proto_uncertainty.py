"""
Prototype-margin uncertainty vs MC Dropout, on the SAME trained prototype model.
Reads the cat1_proto_classifier checkpoint, runs the test set, and computes:
  - prototype-margin uncertainty = (best competing-class sim) - (best own-class sim)
  - MC Dropout uncertainty       = std over 30 dropout-on passes
Then compares them on: error-detection AUROC, abstention curves, and correlation.
Outputs table + plots to experiments/test_results/proto_uncertainty/.
Uses only numpy, pandas, torch, matplotlib, sklearn (no scipy).
"""
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
OUT = os.path.expanduser("~/protoecgnet/experiments/test_results/proto_uncertainty")
os.makedirs(OUT, exist_ok=True)


def spearman(a, b):
    """Spearman correlation = Pearson correlation of ranks (numpy only)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))


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


# ---- load model ----
ckpt = best_ckpt(CKPT_DIR)
print("Using checkpoint:", ckpt)
lm = load_label_mappings(custom_groups=True, prototype_category=1)
num_classes = len(lm["custom"])
_, _, test_loader, _ = get_dataloaders(batch_size=32, mode="1D", sampling_rate=100,
    label_set="1", work_num=4, return_sample_ids=False, custom_groups=True,
    standardize=False, remove_baseline=True)
model = ProtoECGNet1D(num_classes=num_classes, single_class_prototype_per_class=6,
    joint_prototypes_per_border=0, proto_dim=512, backbone="resnet1d18",
    prototype_activation_function="log", latent_space_type="arc",
    add_on_layers_type="linear", class_specific=True, last_layer_connection_weight=1.0,
    m=0.05, dropout=0.35, custom_groups=True, label_set="1",
    pretrained_weights=ckpt).to(DEVICE)
model.eval()
proto_identity = model.prototype_class_identity.detach().cpu().numpy()  # (P, C)
print("prototype_class_identity:", proto_identity.shape)

# ---- deterministic pass: prototype activations, probs, labels ----
acts, probs, labels = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        logits, _, activations = model(x.to(DEVICE))
        acts.append(activations.cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
acts = np.concatenate(acts); probs = np.concatenate(probs)
labels = np.concatenate(labels).astype(float)
print("acts", acts.shape, "probs", probs.shape, "labels", labels.shape)

# ---- prototype-margin uncertainty ----
same = proto_identity.T  # (C, P): row c marks prototypes of class c
support = np.zeros_like(probs); competition = np.zeros_like(probs)
for c in range(num_classes):
    mask = same[c] == 1
    support[:, c] = acts[:, mask].max(axis=1)
    competition[:, c] = acts[:, ~mask].max(axis=1)
margin = support - competition
unc_margin = -np.abs(margin)            # small |margin| = near boundary = most uncertain

# ---- MC Dropout on the SAME model ----
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
        mc.append(np.concatenate(p))
        print(f"  MC pass {t+1}/{T_MC}")
mc = np.stack(mc)
mc_mean, mc_std = mc.mean(0), mc.std(0)
unc_mc = mc_std

# ---- predictions / errors (from deterministic probs) ----
pred = (probs > 0.5).astype(int)
error = (pred != labels).astype(int)
e = error.ravel()

# a) error-detection AUROC
auc_margin = roc_auc_score(e, unc_margin.ravel())
auc_mc = roc_auc_score(e, unc_mc.ravel())
print(f"\nError-detection AUROC: margin={auc_margin:.4f}  MC-dropout={auc_mc:.4f}")

# c) correlation between the two uncertainties
rho = spearman(unc_margin.ravel(), unc_mc.ravel())
print(f"Spearman correlation(margin, MC): {rho:.4f}")

# b) selective prediction (same probs, different ranking)
p_flat, y_flat = probs.ravel(), labels.ravel()
N = len(p_flat)
rows = []
for frac in [0.0, 0.05, 0.10, 0.20, 0.30]:
    k = int(round((1 - frac) * N)); cov = 100 * k / N
    om = np.argsort(unc_margin.ravel())[:k]
    oc = np.argsort(unc_mc.ravel())[:k]
    am, fm, acm = metrics_on(p_flat[om], y_flat[om])
    amc, fmc, acmc = metrics_on(p_flat[oc], y_flat[oc])
    ra, rf = [], []
    for s in range(10):
        ridx = np.random.default_rng(s).permutation(N)[:k]
        a2, f2, _ = metrics_on(p_flat[ridx], y_flat[ridx]); ra.append(a2); rf.append(f2)
    rows.append(dict(abstain=int(frac*100), coverage=round(cov, 1),
        AUROC_margin=round(am, 4), Fmax_margin=round(fm, 4), Acc_margin=round(acm, 4),
        AUROC_mc=round(amc, 4), Fmax_mc=round(fmc, 4), Acc_mc=round(acmc, 4),
        AUROC_rand=round(float(np.mean(ra)), 4), Fmax_rand=round(float(np.mean(rf)), 4)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "proto_vs_mc_selective.csv"), index=False)
print("\n" + df.to_string(index=False))

# ---- plots ----
plt.figure(figsize=(7, 5))
plt.plot(df.coverage, df.AUROC_margin, "o-", label="AUROC (prototype-margin)")
plt.plot(df.coverage, df.AUROC_mc, "s-", label="AUROC (MC dropout)")
plt.plot(df.coverage, df.AUROC_rand, "^--", color="gray", label="AUROC (random)")
plt.gca().invert_xaxis(); plt.xlabel("Coverage %"); plt.ylabel("AUROC on retained")
plt.title("Selective prediction: prototype-margin vs MC dropout")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "selective_auroc.png"), dpi=140); plt.close()

plt.figure(figsize=(7, 5))
plt.plot(df.coverage, df.Fmax_margin, "o-", label="Fmax (prototype-margin)")
plt.plot(df.coverage, df.Fmax_mc, "s-", label="Fmax (MC dropout)")
plt.plot(df.coverage, df.Fmax_rand, "^--", color="gray", label="Fmax (random)")
plt.gca().invert_xaxis(); plt.xlabel("Coverage %"); plt.ylabel("Fmax on retained")
plt.title("Selective prediction: prototype-margin vs MC dropout")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "selective_fmax.png"), dpi=140); plt.close()

idx = np.random.default_rng(0).choice(N, size=min(5000, N), replace=False)
plt.figure(figsize=(6, 6))
plt.scatter(unc_margin.ravel()[idx], unc_mc.ravel()[idx], s=4, alpha=0.3)
plt.xlabel("Prototype-margin uncertainty (competition - support)")
plt.ylabel("MC dropout uncertainty (std)")
plt.title(f"Correlation of the two uncertainties (Spearman {rho:.2f})")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "uncertainty_correlation.png"), dpi=140); plt.close()

np.savez(os.path.join(OUT, "proto_uncertainty_outputs.npz"),
    unc_margin=unc_margin, unc_mc=unc_mc, probs=probs, labels=labels,
    support=support, competition=competition, mc_mean=mc_mean)
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write(f"Checkpoint: {ckpt}\n")
    f.write(f"Error-detection AUROC: prototype-margin={auc_margin:.4f}  MC-dropout={auc_mc:.4f}\n")
    f.write(f"Spearman corr(margin, MC): {rho:.4f}\n\n")
    f.write(df.to_string(index=False) + "\n")
print("\nSaved table, plots, npz, summary to:", OUT)