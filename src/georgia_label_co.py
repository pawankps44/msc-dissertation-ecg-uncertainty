import os, numpy as np, torch
from ecg_utils_georgia import get_georgia_dataloaders, GEORGIA_CLASSES
tr, _, _, _ = get_georgia_dataloaders(return_sample_ids=False)
Y = np.concatenate([y.numpy() for _, y in tr]).astype(np.float32)
inter = Y.T @ Y
den = np.clip(Y.sum(0, keepdims=True) + Y.sum(0, keepdims=True).T - inter, 1e-6, None)
cooc = inter / den
out = os.path.expanduser("~/protoecgnet/experiments/preprocessing/label_cooccur_georgia.pt")
torch.save(torch.tensor(cooc, dtype=torch.float32), out)
print("saved", out, "shape", cooc.shape, "| classes:", GEORGIA_CLASSES)
