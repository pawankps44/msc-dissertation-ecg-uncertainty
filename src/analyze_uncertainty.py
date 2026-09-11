
import os, glob

import numpy as np                 

import pandas as pd                

import matplotlib                  

matplotlib.use("Agg")             

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score   # standard AUROC metric


TEST_DIR = os.path.expanduser("~/protoecgnet/experiments/test_results")          

NPZ      = os.path.join(TEST_DIR, "mc_dropout", "mc_dropout_outputs.npz")        

OUT      = os.path.join(TEST_DIR, "uncertainty_analysis")                        

os.makedirs(OUT, exist_ok=True)                                                  


def load_csv(job):

    files = sorted(glob.glob(os.path.join(TEST_DIR, job, f"{job}_test_results_v*.csv")))

    if not files:

        raise FileNotFoundError(f"No test CSV for {job}")

    df = pd.read_csv(files[-1])                                                   

    lab  = df[[c for c in df.columns if c.startswith("Label_")]].values.astype(float)   

    prob = df[[c for c in df.columns if c.startswith("Prob_")]].values.astype(float)    

    return prob, lab                                                             


def ece_brier(probs, labels, n_bins=10):

    p = probs.ravel(); y = labels.ravel()          # flatten everything to one long list

    brier = float(np.mean((p - y) ** 2))           # Brier score

    edges = np.linspace(0, 1, n_bins + 1)          # split [0,1] into 10 confidence "bins"

    ece, N = 0.0, len(p)

    for i in range(n_bins):                        

        lo, hi = edges[i], edges[i + 1]

        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)    

        if m.sum() > 0:


            ece += (m.sum() / N) * abs(p[m].mean() - y[m].mean())

    return float(ece), brier


def macro_auc(probs, labels):

    aucs = [roc_auc_score(labels[:, i], probs[:, i])

            for i in range(labels.shape[1]) if len(np.unique(labels[:, i])) > 1]

    return float(np.mean(aucs)) if aucs else float("nan")


def fmax(probs, labels):

    best = 0.0

    for thr in np.linspace(0.02, 0.98, 49):        # try 49 thresholds from 0.02 to 0.98

        pred = (probs >= thr).astype(int)          # turn probs into 0/1 at this threshold

        tp = (pred * labels).sum()                 # true positives

        fp = (pred * (1 - labels)).sum()           # false positives

        fn = ((1 - pred) * labels).sum()           # false negatives

        prec = tp / (tp + fp + 1e-9)               # precision (1e-9 avoids divide-by-zero)

        rec  = tp / (tp + fn + 1e-9)               # recall

        best = max(best, 2 * prec * rec / (prec + rec + 1e-9))   # F1, keep the best

    return float(best)


# A perfectly calibrated model sits on the diagonal (x == y).

def reliability_points(probs, labels, n_bins=10):

    p = probs.ravel(); y = labels.ravel()

    edges = np.linspace(0, 1, n_bins + 1); xs, ys = [], []

    for i in range(n_bins):

        lo, hi = edges[i], edges[i + 1]

        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)

        if m.sum() > 0:

            xs.append(p[m].mean()); ys.append(y[m].mean())

    return np.array(xs), np.array(ys)



prob_base,  y_base  = load_csv("cat1_feat_baseline")   

prob_focal, y_focal = load_csv("cat1_feat_focal")      

mc = np.load(NPZ)                                       # MC Dropout results

prob_mc, std_mc, y_mc = mc["mean"], mc["std"], mc["labels"].astype(float)


print("\n=== 1) CALIBRATION (ECE / Brier) ===")

rows = []

for name, p, y in [("Baseline (BCE)", prob_base, y_base),

                   ("Focal loss", prob_focal, y_focal),

                   ("MC Dropout (mean)", prob_mc, y_mc)]:

    ece, brier = ece_brier(p, y)                   # calibration error

    auc = macro_auc(p, y)                          # overall accuracy

    rows.append((name, auc, ece, brier))

    print(f"  {name:20s}  AUROC={auc:.4f}  ECE={ece:.4f}  Brier={brier:.4f}")

pd.DataFrame(rows, columns=["model","AUROC","ECE","Brier"]).to_csv(

    os.path.join(OUT, "calibration_table.csv"), index=False)           


plt.figure(figsize=(6, 6))

plt.plot([0, 1], [0, 1], "k--", label="perfect")                       # dashed = perfectly calibrated

for name, p, y in [("Baseline", prob_base, y_base),

                   ("Focal", prob_focal, y_focal),

                   ("MC Dropout", prob_mc, y_mc)]:

    xs, ys = reliability_points(p, y)

    plt.plot(xs, ys, "o-", label=name)                                 # each model's calibration curve

plt.xlabel("Mean predicted probability"); plt.ylabel("Empirical fraction positive")

plt.title("Reliability diagram"); plt.legend(); plt.tight_layout()

plt.savefig(os.path.join(OUT, "reliability.png"), dpi=140); plt.close()



print("\n=== 2) DO ERRORS HAVE HIGHER UNCERTAINTY? ===")

pred_mc = (prob_mc > 0.5).astype(int)              # turn MC probs into 0/1 predictions

err = (pred_mc != y_mc).astype(int)                # 1 where the prediction is WRONG

unc = std_mc                                       # the uncertainty value

# If uncertainty is meaningful, it should be HIGH exactly where err==1.

# We test that with an AUROC: can uncertainty alone separate right from wrong?

print(f"  Uncertainty detecting errors  AUROC = {roc_auc_score(err.ravel(), unc.ravel()):.4f}  (0.5=useless, >0.5=useful)")

# Label every prediction as TP/FP/FN/TN, then compare their uncertainty.

# TP/TN = correct, FP/FN = errors. We expect errors to be MORE uncertain.

cat = np.empty(err.shape, dtype=object)

cat[(pred_mc == 1) & (y_mc == 1)] = "TP"           # predicted yes, truly yes  (correct)

cat[(pred_mc == 1) & (y_mc == 0)] = "FP"           # predicted yes, truly no   (error)

cat[(pred_mc == 0) & (y_mc == 1)] = "FN"           # predicted no,  truly yes  (error)

cat[(pred_mc == 0) & (y_mc == 0)] = "TN"           # predicted no,  truly no   (correct)

groups = {k: unc.ravel()[cat.ravel() == k] for k in ["TP","FP","FN","TN"]}

for k in ["TP","TN","FP","FN"]:

    print(f"  {k}: n={len(groups[k]):7d}  mean uncertainty={groups[k].mean():.4f}")

corr  = unc.ravel()[err.ravel() == 0]              # uncertainty of all CORRECT predictions

wrong = unc.ravel()[err.ravel() == 1]              # uncertainty of all WRONG predictions

print(f"  CORRECT mean unc={corr.mean():.4f}   INCORRECT mean unc={wrong.mean():.4f}")

# box plot of uncertainty for TP/TN/FP/FN

plt.figure(figsize=(7, 5))

plt.boxplot([groups["TP"], groups["TN"], groups["FP"], groups["FN"]],

            labels=["TP","TN","FP","FN"], showfliers=False)

plt.ylabel("MC Dropout uncertainty (std)")

plt.title("Uncertainty by prediction type (errors = FP, FN)")

plt.tight_layout(); plt.savefig(os.path.join(OUT, "uncertainty_vs_error.png"), dpi=140); plt.close()



print("\n=== 3) ABSTENTION (remove most uncertain) ===")

sample_unc = std_mc.mean(axis=1)                   # one uncertainty number per ECG (avg over classes)

order = np.argsort(sample_unc)                     # sort ECGs from MOST confident -> LEAST confident

abst_rows = []

for cov in [1.00, 0.90, 0.80, 0.70]:               # keep 100%, 90%, 80%, 70% (abstain on the rest)

    keep = order[:int(round(cov * len(order)))]    # the most-confident X% of ECGs

    a  = macro_auc(prob_mc[keep], y_mc[keep])       # AUROC on the kept ones

    fm = fmax(prob_mc[keep], y_mc[keep])            # Fmax on the kept ones

    abst_rows.append((int(cov*100), 100-int(cov*100), a, fm))

    print(f"  coverage {int(cov*100):3d}% (abstain {100-int(cov*100):2d}%)  AUROC={a:.4f}  Fmax={fm:.4f}")

ab = pd.DataFrame(abst_rows, columns=["coverage_%","abstain_%","AUROC","Fmax"])

ab.to_csv(os.path.join(OUT, "abstention_table.csv"), index=False)

#  abstention curve: performance should RISE as we keep fewer, more-confident cases 

plt.figure(figsize=(7, 5))

plt.plot(ab["coverage_%"], ab["AUROC"], "o-", label="AUROC")

plt.plot(ab["coverage_%"], ab["Fmax"],  "s-", label="Fmax")

plt.gca().invert_xaxis()                            # show 100% -> 70% left to right

plt.xlabel("Coverage % kept"); plt.ylabel("Performance on retained")

plt.title("Abstention curve"); plt.legend(); plt.tight_layout()

plt.savefig(os.path.join(OUT, "abstention.png"), dpi=140); plt.close()

print("\nSaved to:", OUT)