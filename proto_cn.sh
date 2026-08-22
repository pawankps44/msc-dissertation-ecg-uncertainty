#!/bin/bash
#SBATCH -J cat1_proto_cn
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python proto_uncertainty_cn.py