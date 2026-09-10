#!/bin/bash
#SBATCH -J cat1_feat_mcdrop2
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general

source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src

python main.py \
  --job_name cat1_feat_mcdrop2 \
  --epochs 50 \
  --batch_size 32 \
  --training_stage feature_extractor \
  --dimension 1D \
  --backbone resnet1d18 \
  --sampling_rate 100 \
  --label_set 1 \
  --custom_groups True \
  --num_workers 4 \
  --patience 10 \
  --dropout 0.3 \
  --checkpoint_dir ~/protoecgnet/experiments/checkpoints \
  --log_dir ~/protoecgnet/experiments/logs \
  --test_dir ~/protoecgnet/experiments/test_results \
  --seed 42