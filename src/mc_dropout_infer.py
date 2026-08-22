"""MC Dropout inference for the Branch-1 (arrhythmia) feature-extractor."""
import os, re, glob, argparse
import numpy as np, torch, torch.nn as nn, pandas as pd
from sklearn.metrics import roc_auc_score
from ecg_utils import get_dataloaders, load_label_mappings
from backbones import (resnet1d18, resnet1d34, resnet1d50, resnet1d101, resnet1d152)

def find_best_ckpt(path):
    if os.path.isfile(path):
        return path
    ckpts = [f for f in glob.glob(os.path.join(path, "*.ckpt")) if "last" not in os.path.basename(f)]
    best, best_auc = None, -1.0
    for f in ckpts:
        m = re.search(r"val_auc=([0-9]+\.[0-9]+)", os.path.basename(f))
        if m and float(m.group(1)) > best_auc:
            best_auc, best = float(m.group(1)), f
    if best is None:
        last = os.path.join(path, "last.ckpt")
        best = last if os.path.exists(last) else None
    return best

def enable_mc_dropout(model):
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(); n += 1
    print(f"Enabled MC Dropout on {n} dropout layer(s).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="resnet1d18")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n_passes", type=int, default=30)
    ap.add_argument("--label_set", default="1")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", default="~/protoecgnet/experiments/test_results/mc_dropout")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.expanduser(args.out_dir); os.makedirs(out_dir, exist_ok=True)

    _, _, test_loader, _ = get_dataloaders(
        batch_size=args.batch_size, mode="1D", sampling_rate=100,
        label_set=args.label_set, work_num=args.num_workers,
        custom_groups=True, standardize=False, remove_baseline=True)
    lm = load_label_mappings(custom_groups=True, prototype_category=int(args.label_set))
    num_classes = len(lm["custom"]); print(f"num_classes = {num_classes}")

    model = eval(args.backbone)(num_classes=num_classes, dropout=args.dropout)
    ckpt_path = find_best_ckpt(os.path.expanduser(args.ckpt)); print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    new_sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    model.to(device); enable_mc_dropout(model)

    all_mean, all_std, all_labels = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            passes = [torch.sigmoid(model(x)).unsqueeze(0) for _ in range(args.n_passes)]
            P = torch.cat(passes, dim=0)
            all_mean.append(P.mean(0).cpu().numpy())
            all_std.append(P.std(0).cpu().numpy())
            all_labels.append(y.numpy())
    mean = np.concatenate(all_mean); std = np.concatenate(all_std); labels = np.concatenate(all_labels)

    aucs = [roc_auc_score(labels[:, i], mean[:, i]) for i in range(labels.shape[1])
            if len(np.unique(labels[:, i])) > 1]
    macro_auc = float(np.mean(aucs))
    print("\n================ MC DROPOUT RESULTS ================")
    print(f"passes per ECG      : {args.n_passes}")
    print(f"macro AUROC (mean)  : {macro_auc:.4f}   (baseline ~0.909)")
    print(f"mean uncertainty std: {std.mean():.4f}")
    print("====================================================")
    np.savez(os.path.join(out_dir, "mc_dropout_outputs.npz"), mean=mean, std=std, labels=labels)
    pd.DataFrame({"mean_uncertainty": std.mean(axis=1)}).to_csv(
        os.path.join(out_dir, "mc_dropout_per_sample.csv"), index=False)
    print("saved ->", out_dir)

if __name__ == "__main__":
    main()