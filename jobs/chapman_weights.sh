#!/bin/bash
#SBATCH -J chap_wts
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python chapman_train_weights.py
