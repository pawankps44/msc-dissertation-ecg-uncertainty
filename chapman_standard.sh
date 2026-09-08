#!/bin/bash
#SBATCH -J chap_std
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
CK=$HOME/protoecgnet/experiments/checkpoints
LG=$HOME/protoecgnet/experiments/logs
TS=$HOME/protoecgnet/experiments/test_results
BASE="--dimension 1D --backbone resnet1d18 --sampling_rate 100 --label_set chapman --custom_groups True --dataset chapman --dropout 0.35 --num_workers 4 --save_weights True --checkpoint_dir $CK --log_dir $LG --test_dir $TS --seed 42"
PROTO="--single_class_prototype_per_class 6 --joint_prototypes_per_border 0 --proto_time_len 32 --proto_dim 512"
python main.py --job_name chapman_feat --training_stage feature_extractor --epochs 50 --batch_size 32 --patience 10 $BASE
FEAT=$(python bestck.py $CK/chapman_feat)
python main.py --job_name chapman_standard_joint --training_stage joint --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO --pretrained_weights "$FEAT"
JCK=$(python bestck.py $CK/chapman_standard_joint)
python main.py --job_name chapman_standard_proj --training_stage projection --epochs 1 --batch_size 32 $BASE $PROTO --pretrained_weights "$JCK"
python main.py --job_name chapman_standard_clf --training_stage classifier --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO --pretrained_weights "$CK/chapman_standard_proj/chapman_standard_proj_projection.pth"
echo "CHAPMAN STANDARD DONE"
