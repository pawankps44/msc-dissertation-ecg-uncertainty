#!/bin/bash
#SBATCH -J pipe
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
SEED=$1
MODE=$2
TAG=${MODE}_s${SEED}
CK=$HOME/protoecgnet/experiments/checkpoints
LG=$HOME/protoecgnet/experiments/logs
TS=$HOME/protoecgnet/experiments/test_results
FEAT=$(python bestck.py $CK/cat1_feat_baseline)
WF=""
if [ "$MODE" = "bce" ]; then WF="--sample_weights_path $HOME/protoecgnet/experiments/preprocessing/train_uncertainty_weights.npz --weight_bce True"; fi
COM="--dimension 1D --backbone resnet1d18 --single_class_prototype_per_class 6 --joint_prototypes_per_border 0 --proto_time_len 32 --proto_dim 512 --sampling_rate 100 --label_set 1 --custom_groups True --dropout 0.35 --num_workers 4 --save_weights True --checkpoint_dir $CK --log_dir $LG --test_dir $TS --seed $SEED"

python main.py --job_name ${TAG}_joint --epochs 100 --batch_size 32 --lr 0.0001 --training_stage joint --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $COM $WF --pretrained_weights "$FEAT"
JCK=$(python bestck.py $CK/${TAG}_joint)
python main.py --job_name ${TAG}_proj --epochs 1 --batch_size 32 --training_stage projection $COM --pretrained_weights "$JCK"
python main.py --job_name ${TAG}_clf --epochs 100 --batch_size 32 --lr 0.0001 --training_stage classifier --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $COM --pretrained_weights "$CK/${TAG}_proj/${TAG}_proj_projection.pth"
echo "PIPELINE DONE for $TAG"
