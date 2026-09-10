#!/bin/bash
#SBATCH -J cat1_proj_wbce
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
python main.py \
  --job_name cat1_proto_proj_wbce \
  --epochs 1 \
  --batch_size 32 \
  --training_stage projection \
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
  --num_workers 4 \
  --save_weights True \
  --checkpoint_dir ~/protoecgnet/experiments/checkpoints \
  --log_dir ~/protoecgnet/experiments/logs \
  --test_dir ~/protoecgnet/experiments/test_results \
  --seed 42 \
  --pretrained_weights "$HOME/protoecgnet/experiments/checkpoints/cat1_proto_joint_wbce/epoch=18-val_auc=0.8337.ckpt"
