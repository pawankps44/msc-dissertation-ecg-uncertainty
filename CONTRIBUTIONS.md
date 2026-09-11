# Contributions

What in this repository is my own work, and what is inherited from the base
model (ProtoECGNet). The base architecture and its training procedure are used
as published; my contribution is the uncertainty, calibration, confidence-
weighting and cross-dataset evaluation built around it, plus the two extra
dataset loaders.

## My own files (created for this project)

**Dataset loaders (Chapman and Georgia)**
- `src/ecg_utils_chapman.py`, `src/ecg_utils_georgia.py` - load and preprocess the
  two external datasets, mirroring the PTB-XL loader's interface.
- `src/chapman_label_co.py`, `src/georgia_label_co.py` - build the label
  co-occurrence matrices for those datasets.

**Uncertainty and analysis**
- `src/mc_dropout_infer.py` - Monte Carlo Dropout inference (30 passes).
- `src/analyze_uncertainty.py` - ECE/Brier, error-vs-uncertainty, abstention.
- `src/selective_prediction.py` - selective prediction / abstention curves.
- `src/proto_uncertainty.py` - prototype-margin uncertainty and MC comparison.
- `src/proto_uncertainty_cn.py` - class-normalised margin (the negative result).
- `src/prototype_consistency.py` - prototype purity.
- `src/proj_lowunc.py` - projection restricted to low-uncertainty examples.

**Confidence-weighted training**
- `src/train_uncertainty_weights.py` - per-sample MC-dropout weights (PTB-XL).
- `src/chapman_train_weights.py`, `src/georgia_train_weights.py` - the same for the
  external datasets.
- `src/weighting_eval.py` - evaluate the five weighting variants.

**Calibration**
- `src/calibration_eval.py`, `src/calibration_eval_s123.py`, `src/calibration_eval_s7.py`
  (PTB-XL), and `src/calibration_eval_chapman*.py`, `src/calibration_eval_georgia*.py`
  - temperature scaling and the four-way calibration/abstention tables.

**Utilities and orchestration**
- `src/bestck.py` - selects the best checkpoint of a run by validation score.
- `run.sh` - the unified runner: `run.sh <dataset> <experiment>` for all three datasets.
- `jobs/` - all the SLURM job scripts (`baseline.sh`, `chapman_*.sh`, `georgia_*.sh`, etc.).
- `archive/` - one-off setup/patch scripts used while wiring the datasets in.

## Base files I modified (ProtoECGNet, with my changes noted)

- `src/backbones.py` - **added a dropout layer** to the 1D ResNet so MC Dropout works.
- `src/main.py` - **added** a `--dataset` argument (PTB-XL / Chapman / Georgia), a
  `--loss_type` switch with focal loss, and the per-sample weighting path.
- `src/training_functions.py` - **added** the focal loss, the per-sample weighting in
  the loss, and class names for the new datasets.
- `src/proto_models1D.py` - **added** the label co-occurrence paths for the new
  datasets and support for the per-sample uncertainty weighting.
- `src/ecg_utils.py` - updated dataset/standardisation paths for my environment.
- `src/push.py` - minor changes to prototype projection/logging.

## Base files used unchanged

- `src/proto_models2D.py`, `src/fusion.py`, `src/inference_fusion.py`, `src/tune.py`,
  `src/label_co.py`, `src/case_explanations.ipynb` - inherited from ProtoECGNet;
  not central to this project's contribution.

## The changes that are genuinely mine 

1. Made MC Dropout possible by adding a dropout layer to the backbone.
2. Made the training loss support per-sample confidence weighting, which enabled the
   confidence-weighted (BCE) result and the loss-scale finding.
3. Added a dataset argument and two new dataset loaders, so the same pipeline runs on
   three cohorts, kept isolated from each other.
4. Wrote all the uncertainty, calibration and abstention analysis, and the unified
   `run.sh` runner.

Everything from the ProtoECGNet classifier itself is credited to its authors
(Sethi et al., MLHC 2025); my layer is the uncertainty, calibration, and the
evaluation and cross-dataset study around it.
