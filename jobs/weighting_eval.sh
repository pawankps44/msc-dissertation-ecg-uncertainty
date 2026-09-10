#!/bin/bash
#SBATCH -J cat1_weval
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python weighting_eval.py
