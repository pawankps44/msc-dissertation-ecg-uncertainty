"""
per-seed calibration error (ECE) for PTB-XL and Chapman,
rebuilt from the nine calibration CSVs so it matches Table 5.5 exactly.

"""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = os.path.expanduser("~/protoecgnet/experiments/test_results")
OUTDIR = os.path.expanduser("~/protoecgnet/experiments/eval_runs")

def seed_of(d):
    if d.endswith("_s7"):   return 7
    if d.endswith("_s123"): return 123
    return 42

def is_dataset(d, ds):
    if ds == "ptbxl":
        return d in ("calibration_eval", "calibration_eval_s7", "calibration_eval_s123")
    return ds in d          # "chapman" / "georgia" appear in the dir name

def keyof(m):
    m = m.lower()
    return ("bce" if "bce" in m else "std") + ("_cal" if "cal" in m else "_unc")

keys   = ["std_unc", "std_cal", "bce_unc", "bce_cal"]
labels = {"std_unc": "Standard", "std_cal": "Standard + cal", "bce_unc": "BCE", "bce_cal": "BCE + cal"}
colors = {"std_unc": "#bdbdbd", "std_cal": "#66bb6a", "bce_unc": "#757575", "bce_cal": "#2e7d32"}

def collect(ds):
    data = {}
    for csv in glob.glob(os.path.join(ROOT, "calibration_eval*", "*.csv")):
        d = os.path.basename(os.path.dirname(csv))
        if not is_dataset(d, ds):
            continue
        s = seed_of(d); data.setdefault(s, {})
        for _, r in pd.read_csv(csv).iterrows():
            data[s][keyof(str(r["model"]))] = float(r["ECE"])
    return data

def plot(ds, title, fname):
    data  = collect(ds)
    seeds = [s for s in (42, 7, 123) if s in data]
    x = np.arange(len(seeds)); w = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, k in enumerate(keys):
        vals = [data[s].get(k, np.nan) for s in seeds]
        b = ax.bar(x + (i - 1.5) * w, vals, w, label=labels[k],
                   color=colors[k], edgecolor="black", linewidth=0.4)
        ax.bar_label(b, fmt="%.3f", fontsize=6, padding=1)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("ECE (lower = better)"); ax.set_title(title)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    out = os.path.join(OUTDIR, fname)
    plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)

plot("ptbxl",   "Calibration error per seed on PTB-XL",  "seed_ece_ptbxl.png")
plot("chapman", "Calibration error per seed on Chapman", "seed_ece_chapman.png")
