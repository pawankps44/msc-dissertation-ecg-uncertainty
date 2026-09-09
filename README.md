# Uncertainty-Aware Arrhythmia Detection from ECG Signals

MSc dissertation project (University of Nottingham). This repository extends
**ProtoECGNet**, an interpretable prototype-based ECG classifier, with
**uncertainty estimation, calibration, and abstention**, and evaluates the result
across three independent hospital datasets and multiple random seeds.

## What this project adds

Starting from ProtoECGNet Branch-1 (rhythm classification), this project adds:

- Monte Carlo Dropout uncertainty, plus an interpretable prototype-margin uncertainty.
- A confidence-weighted (BCE) training variant.
- Post-hoc temperature scaling for probability calibration.
- Error-detection and selective-prediction (abstention) evaluation.
- Validation across **PTB-XL** (Germany), **Chapman-Shaoxing** (China), and
  **Georgia** (Emory, PhysioNet/CinC 2021), each with three seeds.

Headline result: post-hoc temperature scaling reliably improves calibration on all
three datasets at no cost to discrimination or to the useful uncertainty;
confidence-weighting consistently improves error detection, though its own
calibration benefit is dataset-dependent.

## Quick start

Everything runs through one script, choosing the dataset by argument:

```bash
sbatch run.sh <dataset> <experiment>
#   dataset    : ptbxl | chapman | georgia
#   experiment : prep | standard | bce | calib
```

Examples:

```bash
sbatch run.sh georgia prep       # build the label co-occurrence matrix
sbatch run.sh georgia standard   # train the standard pipeline (feat -> joint -> proj -> clf)
sbatch run.sh chapman bce        # confidence-weighted (BCE) pipeline
sbatch run.sh ptbxl  calib       # calibration + abstention evaluation
```

Outputs land under `experiments/`:

- `checkpoints/<job>/` - trained models
- `test_results/<job>/` - metrics and calibration CSVs
- console log - `run_<jobid>.out`

Job names carry the dataset (`georgia_standard_clf`, `chapman_bce_clf`, ...), so
nothing collides across datasets.

## Repository layout

- `run.sh` - unified runner (one command per dataset).
- `src/`
  - `main.py` - training/evaluation entry point; selects the dataset via `--dataset`.
  - `ecg_utils.py`, `ecg_utils_chapman.py`, `ecg_utils_georgia.py` - per-dataset loaders.
  - `proto_models1D.py`, `backbones.py`, `training_functions.py`, `push.py` - model and training.
  - uncertainty / calibration: `mc_dropout_infer.py`, `analyze_uncertainty.py`,
    `selective_prediction.py`, `proto_uncertainty.py`, `proto_uncertainty_cn.py`,
    `train_uncertainty_weights.py`, `weighting_eval.py`, `prototype_consistency.py`,
    `calibration_eval*.py`, `bestck.py`.
- `experiments/` - checkpoints, logs, results, preprocessing (gitignored).
- `archive/` - one-off setup and patch scripts, kept for history.

## Environment

Python 3.10 with PyTorch and PyTorch Lightning, run on a SLURM GPU cluster. See
`environment.yml` / `requirements_cluster.txt`. Activate the virtual environment
before running (for example `source ~/ecg_venv/bin/activate`).

## Datasets

- PTB-XL: https://physionet.org/content/ptb-xl/
- Chapman-Shaoxing and the PhysioNet/Computing in Cardiology 2021 (Georgia) data are
  publicly available on PhysioNet.

Each dataset has its own loader and data folder, and the datasets are kept isolated
so that adding one cannot affect the results of another.

## Base model and credit

This work builds on **ProtoECGNet** by Sethi et al. (MLHC 2025). The interpretable
prototype architecture and its training procedure are theirs; the uncertainty,
calibration, confidence-weighting, cross-dataset evaluation, and the additional
dataset loaders are the contribution of this dissertation.

> Sethi, S., Chen, D., Statchen, T., Burkhart, M. C., Bhandari, N., Ramadan, B., &
> Beaulieu-Jones, B. (2025). ProtoECGNet: Case-Based Interpretable Deep Learning for
> Multi-Label ECG Classification with Contrastive Learning. Proceedings of the 10th
> Machine Learning for Healthcare Conference (MLHC), PMLR 298.
> https://proceedings.mlr.press/v298/sethi25a.html
