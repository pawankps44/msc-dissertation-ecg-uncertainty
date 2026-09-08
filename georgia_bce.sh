#!/bin/bash
#SBATCH -J geo_bce
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o %x_%j.out
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
CK=$HOME/protoecgnet/experiments/checkpoints
BASE="--dimension 1D --backbone resnet1d18 --sampling_rate 100 --label_set georgia --custom_groups True --dataset georgia --dropout 0.35 --num_workers 4 --save_weights True --checkpoint_dir $CK --log_dir $HOME/protoecgnet/experiments/logs --test_dir $HOME/protoecgnet/experiments/test_results --seed 42"
PROTO="--single_class_prototype_per_class 6 --joint_prototypes_per_border 0 --proto_time_len 32 --proto_dim 512"
WF="--sample_weights_path $HOME/protoecgnet/experiments/preprocessing/georgia_train_weights.npz --weight_bce True"
FEAT=$(python bestck.py $CK/georgia_feat)
python main.py --job_name georgia_bce_joint --training_stage joint --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO $WF --pretrained_weights "$FEAT"
JCK=$(python bestck.py $CK/georgia_bce_joint)
python main.py --job_name georgia_bce_proj --training_stage projection --epochs 1 --batch_size 32 $BASE $PROTO --pretrained_weights "$JCK"
python main.py --job_name georgia_bce_clf --training_stage classifier --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO --pretrained_weights "$CK/georgia_bce_proj/georgia_bce_proj_projection.pth"
echo "GEORGIA BCE DONE"
