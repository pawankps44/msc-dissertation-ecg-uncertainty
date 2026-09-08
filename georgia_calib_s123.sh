#!/bin/bash
#SBATCH -J geo_calib_s123
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o %x_%j.out
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python calibration_eval_georgia_s123.py
