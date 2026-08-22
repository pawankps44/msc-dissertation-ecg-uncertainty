#!/bin/bash
#SBATCH -J cat1_clf_wbce
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python main.py \
  --job_name cat1_proto_classifier_wbce \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.0001 \
  --training_stage classifier \
  --dimension 1D \
  --backbone resnet1d18 \
  --single_class_prototype_per_class 6 \
  --joint_prototypes_per_border 0 \
  --proto_time_len 32 \
  --proto_dim 512 \
  --sampling_rate 100 \
  --label_set 1 \
  --custom_groups True \
  --dropout 0.35 \
  --l2 0.00017 \
  --scheduler_type CosineAnnealingLR \
  --num_workers 4 \
  --patience 10 \
  --save_weights True \
  --checkpoint_dir ~/protoecgnet/experiments/checkpoints \
  --log_dir ~/protoecgnet/experiments/logs \
  --test_dir ~/protoecgnet/experiments/test_results \
  --seed 42 \
  --pretrained_weights "$HOME/protoecgnet/experiments/checkpoints/cat1_proto_proj_wbce/cat1_proto_proj_wbce_projection.pth"
