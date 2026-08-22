#!/bin/bash
#SBATCH -J cat1_calib_s7
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python calibration_eval_s7.py
