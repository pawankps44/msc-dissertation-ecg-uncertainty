# =====================================================================
#  selective_prediction.py
#  The abstention experiment: let the model "skip" its most uncertain
#  predictions, and check if it gets more reliable on the ones it keeps.
#  We also compare against RANDOM skipping, to prove the uncertainty is
#  doing real work. Reads saved MC Dropout outputs -> no GPU needed.
# =====================================================================
import os
import numpy as np                 # arrays / maths
import pandas as pd                # for the results table
import matplotlib                  # plotting
matplotlib.use("Agg")             # save plots to file (no screen)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---- where the saved MC Dropout file is, and where to save results ----
TEST_DIR = os.path.expanduser("~/protoecgnet/experiments/test_results")
NPZ = os.path.join(TEST_DIR, "mc_dropout", "mc_dropout_outputs.npz")   # mean, std, labels
OUT = os.path.join(TEST_DIR, "selective_prediction"); os.makedirs(OUT, exist_ok=True)

ABSTAIN = [0.0, 0.05, 0.10, 0.20, 0.30]   # how much to discard (0%,5%,10%,20%,30%)
RANDOM_SEEDS = 10                          # how many random runs to average for the baseline


# ---- Fmax: the best F1 score across all decision thresholds ----
def fmax(p, y):
    best = 0.0
    for thr in np.linspace(0.02, 0.98, 49):    # try 49 thresholds
        pred = (p >= thr).astype(int)          # turn probabilities into yes/no
        tp = (pred * y).sum()                   # true positives
        fp = (pred * (1 - y)).sum()             # false positives
        fn = ((1 - pred) * y).sum()             # false negatives
        prec = tp / (tp + fp + 1e-9)            # precision
        rec  = tp / (tp + fn + 1e-9)            # recall
        best = max(best, 2 * prec * rec / (prec + rec + 1e-9))   # keep best F1
    return best


# ---- compute all the metrics on a given set of predictions ----
def metrics_on(p, y):
    auroc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")  # ranking quality
    pred = (p >= 0.5).astype(int)              # predictions at the 0.5 cutoff
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / (tp + fp + 1e-9)               # precision
    rec  = tp / (tp + fn + 1e-9)               # recall
    acc  = (pred == y).mean()                  # accuracy
    return auroc, fmax(p, y), prec, rec, acc


# ---- load the saved MC Dropout outputs and flatten to one long list ----
# Each entry is one prediction (a single class on a single ECG):
#   p = the averaged probability, u = its uncertainty, y = the true 0/1 label
mc = np.load(NPZ)
p = mc["mean"].ravel()
u = mc["std"].ravel()
y = mc["labels"].astype(float).ravel()
N = len(p)
order = np.argsort(u)              # sort by uncertainty, LOW first (keep confident ones first)
print(f"Total predictions: {N}")

rows = []
for frac in ABSTAIN:
    k = int(round((1 - frac) * N))   # how many predictions we KEEP
    keep = order[:k]                 # the k least-uncertain predictions
    cov = 100.0 * k / N              # coverage = % we still answer

    # --- metrics when we keep the most confident k (uncertainty-based) ---
    au, fm, pr, rc, ac = metrics_on(p[keep], y[keep])

    # --- RANDOM baseline: keep a random k instead, averaged over seeds ---
    # If uncertainty is useful, the line above should beat this one.
    rnd_au, rnd_fm = [], []
    for s in range(RANDOM_SEEDS):
        ridx = np.random.default_rng(s).permutation(N)[:k]   # random k predictions
        a2, f2, *_ = metrics_on(p[ridx], y[ridx])
        rnd_au.append(a2); rnd_fm.append(f2)

    rows.append({"abstain_%": int(frac*100), "coverage_%": round(cov,1),
                 "AUROC": round(au,4), "Fmax": round(fm,4),
                 "Precision": round(pr,4), "Recall": round(rc,4), "Accuracy": round(ac,4),
                 "AUROC_random": round(float(np.mean(rnd_au)),4),
                 "Fmax_random": round(float(np.mean(rnd_fm)),4)})
    print(f"abstain {int(frac*100):2d}%  cov {cov:5.1f}%  "
          f"AUROC {au:.4f} (rand {np.mean(rnd_au):.4f})  "
          f"Fmax {fm:.4f} (rand {np.mean(rnd_fm):.4f})  P {pr:.3f}  R {rc:.3f}  Acc {ac:.4f}")

# ---- save the table ----
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "selective_prediction_table.csv"), index=False)

# ---- plot 1: AUROC + Fmax vs coverage, uncertainty vs random ----
plt.figure(figsize=(7,5))
plt.plot(df["coverage_%"], df["AUROC"], "o-", label="AUROC (uncertainty)")
plt.plot(df["coverage_%"], df["AUROC_random"], "o--", color="gray", label="AUROC (random)")
plt.plot(df["coverage_%"], df["Fmax"], "s-", label="Fmax (uncertainty)")
plt.plot(df["coverage_%"], df["Fmax_random"], "s--", color="lightgray", label="Fmax (random)")
plt.gca().invert_xaxis()           # show 100% on the left, 70% on the right
plt.xlabel("Coverage % (kept after abstaining)"); plt.ylabel("Performance on retained")
plt.title("Selective prediction: uncertainty vs random abstention"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "selective_auroc_fmax.png"), dpi=140); plt.close()

# ---- plot 2: precision / recall / accuracy vs coverage ----
plt.figure(figsize=(7,5))
plt.plot(df["coverage_%"], df["Precision"], "o-", label="Precision")
plt.plot(df["coverage_%"], df["Recall"], "s-", label="Recall")
plt.plot(df["coverage_%"], df["Accuracy"], "^-", label="Accuracy")
plt.gca().invert_xaxis()
plt.xlabel("Coverage % (kept after abstaining)"); plt.ylabel("Performance on retained")
plt.title("Selective prediction: precision / recall / accuracy"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "selective_prec_rec.png"), dpi=140); plt.close()

print("\nSaved table + 2 plots to:", OUT)

