#!/bin/bash
#SBATCH -J geo_prep
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o %x_%j.out
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python georgia_label_co.py
echo "GEORGIA PREP DONE"
