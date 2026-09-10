import glob, os, pandas as pd
root = os.path.expanduser("~/protoecgnet/experiments/test_results")
frames = []
for csv in sorted(glob.glob(os.path.join(root, "calibration_eval*", "*.csv"))):
    df = pd.read_csv(csv); df.insert(0, "run", os.path.basename(os.path.dirname(csv)))
    frames.append(df)
alldf = pd.concat(frames, ignore_index=True)
cols = [c for c in ["run","model","AUROC","Fmax","ECE","Brier","ErrDet","ErrDet_AUROC"] if c in alldf.columns]
print(alldf[cols].to_string(index=False))
alldf.to_csv(os.path.expanduser("~/protoecgnet/experiments/eval_runs/cross_summary.csv"), index=False)
print("\nsaved -> ~/protoecgnet/experiments/eval_runs/cross_summary.csv")