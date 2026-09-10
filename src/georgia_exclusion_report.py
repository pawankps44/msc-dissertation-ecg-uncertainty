"""
D02 report: Georgia exclusion flow, class support, native sampling rates, and
the shared rhythm label spaces. Read-only; just prints counts for the
dissertation (Section 3.1). Run from ~/protoecgnet/src:

    python georgia_exclusion_report.py

The exclusion logic here mirrors get_georgia_dataloaders() exactly, so the KEPT
count should equal the number of records the pipeline actually used.
"""
import os, glob, collections
import numpy as np, wfdb
from ecg_utils_georgia import DATA_DIR, RHYTHM, CODES, GEORGIA_CLASSES, CODE2IDX, _dx

heas = sorted(glob.glob(os.path.join(DATA_DIR, "*", "*.hea")))
n_total = len(heas)

n_no_dx = n_read_fail = n_wrong_shape = n_non_finite = n_kept = 0
support = np.zeros(len(CODES), dtype=int)
fs_counter = collections.Counter()

for h in heas:
    lab = np.zeros(len(CODES)); hit = False
    for code in _dx(h):
        if code in CODE2IDX:
            lab[CODE2IDX[code]] = 1.0; hit = True
    if not hit:                          # no selected rhythm label
        n_no_dx += 1; continue
    rec = h[:-4]
    try:
        sig, fields = wfdb.rdsamp(rec)
    except Exception:                    # unreadable record
        n_read_fail += 1; continue
    if sig.shape != (5000, 12):          # wrong shape / length
        n_wrong_shape += 1; continue
    if not np.isfinite(sig).all():       # corrupt (NaN/Inf) leads
        n_non_finite += 1; continue
    n_kept += 1
    support += lab.astype(int)
    fs_counter[fields.get("fs")] += 1

print("\n==== GEORGIA EXCLUSION FLOW ====")
print(f"{'Total .hea records found':42s} {n_total}")
print(f"{'  excluded: no selected rhythm label':42s} {n_no_dx}")
print(f"{'  excluded: unreadable (rdsamp failed)':42s} {n_read_fail}")
print(f"{'  excluded: shape != (5000,12)':42s} {n_wrong_shape}")
print(f"{'  excluded: non-finite samples':42s} {n_non_finite}")
print(f"{'KEPT (used)':42s} {n_kept}")
assert n_total == n_no_dx + n_read_fail + n_wrong_shape + n_non_finite + n_kept, "buckets do not sum"

print("\n==== PER-CLASS SUPPORT (kept records; multi-label) ====")
for name, s in zip(GEORGIA_CLASSES, support):
    print(f"  {name:5s} {int(s)}")

print("\n==== NATIVE SAMPLING RATE OF KEPT RECORDS ====")
for fs, c in sorted(fs_counter.items(), key=lambda x: -x[1]):
    print(f"  {fs} Hz : {c}")

print("\n==== CHAPMAN/GEORGIA 7 SNOMED RHYTHM CLASSES (shared label space) ====")
for code, name in RHYTHM:
    print(f"  {name:5s} {code}")

# 16 PTB-XL category-1 labels (custom grouping used for Branch-1)
try:
    from ecg_utils import load_label_mappings
    lm = load_label_mappings(custom_groups=True, prototype_category=1)
    cat1 = list(lm["custom"])
    print(f"\n==== PTB-XL CATEGORY-1 LABELS ({len(cat1)}) ====")
    print("  " + ", ".join(map(str, cat1)))
except Exception as ex:
    print("\n[could not load PTB-XL cat-1 labels:", repr(ex), "]")
