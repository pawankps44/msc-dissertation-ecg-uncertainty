import os, glob
import numpy as np, torch
import wfdb
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from ecg_utils import PTBXL_Dataset_1D, remove_baseline_wander  # reused, not modified
DATA_DIR = os.path.expanduser("~/ecg_data/georgia")
# Same 7 rhythm classes as Chapman (identical label space for cross-dataset comparison)
RHYTHM = [("426783006","SR"),("164889003","AFIB"),("426177001","SB"),
          ("427084000","ST"),("164890007","AF"),("426761007","SVT"),
          ("713422000","AT")]
CODES=[c for c,_ in RHYTHM]
GEORGIA_CLASSES=[n for _,n in RHYTHM]
CODE2IDX={c:i for i,c in enumerate(CODES)}
MAX_CLASS_WEIGHT = 30.0   # cap pos-weight so ultra-rare SVT/AT cannot explode the loss
def _dx(hea):
    for line in open(hea):
        if line.replace(" ","").startswith("#Dx:"):
            return [x.strip() for x in line.split(":",1)[1].split(",")]
    return []
def get_georgia_dataloaders(batch_size=32, work_num=4, return_sample_ids=False,
                            remove_baseline=True, seed=42, **kwargs):
    heas=sorted(glob.glob(os.path.join(DATA_DIR,"*","*.hea")))
    X=[]; Y=[]; ids=[]; skipped_nan=0
    for h in heas:
        lab=np.zeros(len(CODES),dtype=np.float32); hit=False
        for code in _dx(h):
            if code in CODE2IDX: lab[CODE2IDX[code]]=1.0; hit=True
        if not hit: continue
        rec=h[:-4]
        try:
            sig,_=wfdb.rdsamp(rec)
        except Exception:
            continue
        if sig.shape!=(5000,12): continue
        if not np.isfinite(sig).all():   # skip corrupt records (NaN/Inf leads) at the source
            skipped_nan+=1; continue
        sig=resample_poly(sig, up=1, down=5, axis=0)   # 500 -> 100 Hz => (1000,12)
        X.append(sig.astype(np.float32)); Y.append(lab); ids.append(int(''.join(ch for ch in os.path.basename(rec) if ch.isdigit())))
    X=np.stack(X); Y=np.stack(Y); ids=np.array(ids)
    print("Georgia loaded:", X.shape, "labels/class:", Y.sum(0).astype(int).tolist(), "| skipped_nan_records:", skipped_nan)
    if remove_baseline:
        X=remove_baseline_wander(X, sampling_rate=100)
    X=np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)   # safety net vs filter-induced NaN
    rng=np.random.default_rng(seed); idx=rng.permutation(len(X))
    ntr=int(0.8*len(X)); nva=int(0.1*len(X))
    tr,va,te=idx[:ntr], idx[ntr:ntr+nva], idx[ntr+nva:]
    c=Y[tr].sum(0); tot=c.sum()
    cw=torch.tensor([(tot-ci)/(ci+1e-6) for ci in c], dtype=torch.float32)
    cw=torch.clamp(cw, max=MAX_CLASS_WEIGHT)   # cap to prevent loss explosion on ultra-rare classes
    def mk(i,sh):
        return DataLoader(PTBXL_Dataset_1D(X[i],Y[i],ids[i],return_sample_ids),
                          batch_size=batch_size, shuffle=sh, num_workers=work_num)
    return mk(tr,True), mk(va,False), mk(te,False), cw
