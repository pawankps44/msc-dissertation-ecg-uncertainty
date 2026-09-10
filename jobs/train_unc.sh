#!/bin/bash
#SBATCH -J cat1_train_unc
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python train_uncertainty_weights.py
