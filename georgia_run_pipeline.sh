#!/bin/bash
#SBATCH -J geo_pipe
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o %x_%j.out
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
SEED=$1; MODE=$2
TAG=georgia_${MODE}_s${SEED}
CK=$HOME/protoecgnet/experiments/checkpoints
BASE="--dimension 1D --backbone resnet1d18 --sampling_rate 100 --label_set georgia --custom_groups True --dataset georgia --dropout 0.35 --num_workers 4 --save_weights True --checkpoint_dir $CK --log_dir $HOME/protoecgnet/experiments/logs --test_dir $HOME/protoecgnet/experiments/test_results --seed $SEED"
PROTO="--single_class_prototype_per_class 6 --joint_prototypes_per_border 0 --proto_time_len 32 --proto_dim 512"
WF=""
if [ "$MODE" = "bce" ]; then WF="--sample_weights_path $HOME/protoecgnet/experiments/preprocessing/georgia_train_weights.npz --weight_bce True"; fi
FEAT=$(python bestck.py $CK/georgia_feat)
python main.py --job_name ${TAG}_joint --training_stage joint --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO $WF --pretrained_weights "$FEAT"
JCK=$(python bestck.py $CK/${TAG}_joint)
python main.py --job_name ${TAG}_proj --training_stage projection --epochs 1 --batch_size 32 $BASE $PROTO --pretrained_weights "$JCK"
python main.py --job_name ${TAG}_clf --training_stage classifier --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO --pretrained_weights "$CK/${TAG}_proj/${TAG}_proj_projection.pth"
echo "GEORGIA ${TAG} DONE"
