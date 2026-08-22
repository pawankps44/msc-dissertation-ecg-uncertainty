import sys, glob, re, os
d = sys.argv[1]
fs = [f for f in glob.glob(os.path.join(d, "*.ckpt")) if "val_auc" in f]
b = max(fs, key=lambda f: float(re.search(r"val_auc=([0-9]+\.[0-9]+)", os.path.basename(f)).group(1)))
print(b)
