#!/bin/bash
#SBATCH -J cat1_pconsist
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python prototype_consistency.py
