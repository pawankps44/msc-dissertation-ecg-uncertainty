#!/bin/bash
#SBATCH -J cat1_mcdrop_infer
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general

source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src

python mc_dropout_infer.py \
  --ckpt ~/protoecgnet/experiments/checkpoints/cat1_feat_mcdrop2 \
  --backbone resnet1d18 \
  --dropout 0.3 \
  --n_passes 30 \
  --label_set 1