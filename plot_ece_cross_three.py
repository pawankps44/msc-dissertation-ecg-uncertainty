import glob, os, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/protoecgnet/experiments/test_results")
OUT  = os.path.expanduser("~/protoecgnet/experiments/eval_runs/ece_cross_three.png")

def dataset_of(dirname):
    if "georgia" in dirname: return "Georgia"
    if "chapman" in dirname: return "Chapman"
    return "PTB-XL"

# key -> list of ECE values across seeds
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for csv in glob.glob(os.path.join(ROOT, "calibration_eval*", "*.csv")):
    ds = dataset_of(os.path.basename(os.path.dirname(csv)))
    df = pd.read_csv(csv)
    for _, r in df.iterrows():
        m = str(r["model"]).lower()
        base = "bce" if "bce" in m else "std"
        state = "cal" if "cal" in m else "unc"
        acc[ds][f"{base}_{state}"].append(float(r["ECE"]))

datasets = ["PTB-XL", "Chapman", "Georgia"]
keys     = ["std_unc", "std_cal", "bce_unc", "bce_cal"]
labels   = {"std_unc": "Standard", "std_cal": "Standard + cal",
            "bce_unc": "BCE", "bce_cal": "BCE + cal"}
colors   = {"std_unc": "#bdbdbd", "std_cal": "#66bb6a",
            "bce_unc": "#757575", "bce_cal": "#2e7d32"}

means = {ds: {k: (np.mean(acc[ds][k]) if acc[ds][k] else np.nan) for k in keys}
         for ds in datasets}

x = np.arange(len(datasets)); w = 0.2
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, k in enumerate(keys):
    vals = [means[ds][k] for ds in datasets]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=labels[k],
                  color=colors[k], edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)

ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("ECE (lower = better)")
ax.set_title("Calibration error across the three datasets (seed-averaged)")
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print("saved figure ->", OUT)
print("\nseed-averaged ECE:")
for ds in datasets:
    print(" ", ds, {k: round(means[ds][k], 3) for k in keys})
