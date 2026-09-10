# Uncertainty-Aware Arrhythmia Detection from ECG Signals

Code for the MSc dissertation *Uncertainty-Aware Arrhythmia Detection from ECG Signals*
(University of Nottingham, 2026). It extends the interpretable **ProtoECGNet** classifier
with **uncertainty estimation, calibration, and abstention**, and validates the result
across three independent ECG datasets and three random seeds.

## Overview

Deep learning reads ECGs accurately but usually returns a single prediction with no
sense of how reliable it is, and most models are black boxes. This project takes an
interpretable, prototype-based classifier (ProtoECGNet) and makes it **aware of its own
uncertainty and well-calibrated**: it estimates uncertainty with Monte Carlo Dropout and
an interpretable prototype-margin, calibrates its probabilities with temperature scaling,
and can **abstain** on the cases it is least sure about. The behaviour is tested on three
datasets from different hospitals to check that it generalises.

![Base model architecture](ProtoECGNet_architecture.jpg)

## Highlights

- **MC-Dropout** uncertainty and an interpretable **prototype-margin** uncertainty.
- A **confidence-weighted (BCE)** training variant, and the *loss-scale* finding that
  explains where confidence weighting can and cannot act.
- **Temperature scaling** for calibration, and **selective prediction / abstention**.
- **Cross-dataset (three cohorts) and multi-seed** validation, reporting what generalises
  and what is dataset-dependent.

## Installation

    conda env create -f environment.yml
    conda activate ecg_env
    # or, on a cluster: pip install -r requirements_cluster.txt

## Data

- **PTB-XL** — https://physionet.org/content/ptb-xl/
- **Chapman-Shaoxing** and the **PhysioNet/Computing in Cardiology 2021 (Georgia)** data
  are publicly available on PhysioNet.

Each dataset has its own loader and data folder; they are kept isolated so that adding
one cannot affect another's results.

## Usage

Everything runs through one script, choosing the dataset by argument:

    sbatch run.sh <dataset> <experiment>
    #   dataset    : ptbxl | chapman | georgia
    #   experiment : prep | standard | bce | calib

    sbatch run.sh georgia prep       # build the label co-occurrence matrix
    sbatch run.sh georgia standard   # train the standard pipeline (feat -> joint -> proj -> clf)
    sbatch run.sh chapman bce        # confidence-weighted (BCE) pipeline
    sbatch run.sh ptbxl  calib       # calibration + abstention evaluation

Trained models and metrics land under `experiments/` (`checkpoints/<job>/`,
`test_results/<job>/`); each job's console log is written to `logs/`.

## Results

Seed-averaged over three seeds. ECE is shown uncalibrated then after temperature scaling
(lower ECE is better).

| Dataset | Std AUROC | Std ECE -> cal | BCE AUROC | BCE ECE -> cal | ErrDet (BCE) |
|---|---|---|---|---|---|
| PTB-XL  | 0.870 | 0.071 -> 0.056 | 0.867 | 0.063 -> 0.046 | 0.921 |
| Chapman | 0.981 | 0.051 -> 0.028 | 0.983 | 0.055 -> 0.033 | 0.937 |
| Georgia | 0.923 | 0.070 -> 0.053 | 0.925 | 0.068 -> 0.047 | 0.893 |

Temperature scaling reduces calibration error on every dataset with no cost to AUROC;
confidence-weighting improves error detection, though its own calibration benefit is
dataset-dependent.

## Repository structure

    run.sh              # unified runner (one command per dataset)
    src/                # model and code (main.py selects the dataset via --dataset)
    jobs/               # SLURM job scripts
    experiments/        # checkpoints, results, preprocessing (gitignored)
    logs/               # job console logs (gitignored)
    archive/            # one-off setup/patch scripts
    CONTRIBUTIONS.md    # what is my own work vs inherited from ProtoECGNet

## Citation

If you use this work, please cite the dissertation:

    @mastersthesis{krishnamoorthy2026uncertainty,
      title  = {Uncertainty-Aware Arrhythmia Detection from ECG Signals},
      author = {Krishnamoorthy, Pawan Subramanyan},
      school = {University of Nottingham},
      year   = {2026}
    }

## Acknowledgements

This work builds on **ProtoECGNet** (Sethi et al., MLHC 2025). The interpretable prototype
architecture and its training procedure are theirs; the uncertainty, calibration,
confidence-weighting, cross-dataset evaluation, and the additional dataset loaders are the
contribution of this project. See `CONTRIBUTIONS.md`.

    @inproceedings{sethi2025protoecgnet,
      title     = {ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG Classification with Contrastive Learning},
      author    = {Sethi, Sahil and Chen, David and Statchen, Thomas and Burkhart, Michael C. and Bhandari, Nipun and Ramadan, Bashar and Beaulieu-Jones, Brett},
      booktitle = {Proceedings of the 10th Machine Learning for Healthcare Conference (MLHC)},
      series    = {PMLR}, volume = {298}, year = {2025}
    }
