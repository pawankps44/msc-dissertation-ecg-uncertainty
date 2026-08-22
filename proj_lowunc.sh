#!/bin/bash
#SBATCH -J cat1_proj_lowunc
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python proj_lowunc.py
