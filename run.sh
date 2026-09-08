#!/bin/bash
#SBATCH -J run
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o %x_%j.out
# Unified runner: sbatch run.sh <dataset> <experiment>
#   dataset    : ptbxl | chapman | georgia
#   experiment : prep | standard | bce | calib
#   e.g.  sbatch run.sh georgia standard   |   sbatch run.sh ptbxl calib
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src

DATASET=$1; EXP=$2
CK=$HOME/protoecgnet/experiments/checkpoints
LG=$HOME/protoecgnet/experiments/logs
TS=$HOME/protoecgnet/experiments/test_results
PP=$HOME/protoecgnet/experiments/preprocessing

case "$DATASET" in
  ptbxl)   DS="--dataset ptbxl --label_set 1 --custom_groups True" ;;
  chapman) DS="--dataset chapman --label_set chapman --custom_groups True" ;;
  georgia) DS="--dataset georgia --label_set georgia --custom_groups True" ;;
  *) echo "Unknown dataset '$DATASET' (use ptbxl|chapman|georgia)"; exit 1 ;;
esac
# per-dataset helper names (PTB-XL keeps its original names)
PREP=${DATASET}_label_co.py;          [ "$DATASET" = "ptbxl" ] && PREP=label_co.py
CALIB=calibration_eval_${DATASET}.py; [ "$DATASET" = "ptbxl" ] && CALIB=calibration_eval.py
FEATDIR=${DATASET}_feat;              [ "$DATASET" = "ptbxl" ] && FEATDIR=cat1_feat_baseline
WEIGHTS=$PP/${DATASET}_train_weights.npz; [ "$DATASET" = "ptbxl" ] && WEIGHTS=$PP/train_uncertainty_weights.npz

BASE="--dimension 1D --backbone resnet1d18 --sampling_rate 100 $DS --dropout 0.35 --num_workers 4 --save_weights True --checkpoint_dir $CK --log_dir $LG --test_dir $TS --seed 42"
PROTO="--single_class_prototype_per_class 6 --joint_prototypes_per_border 0 --proto_time_len 32 --proto_dim 512"

case "$EXP" in
  prep)  python "$PREP" ;;
  calib) python "$CALIB" ;;
  standard|bce)
     TAG=${DATASET}_${EXP}
     WF=""; [ "$EXP" = "bce" ] && WF="--sample_weights_path $WEIGHTS --weight_bce True"
     [ -d "$CK/$FEATDIR" ] || python main.py --job_name $FEATDIR --training_stage feature_extractor --epochs 50 --batch_size 32 --patience 10 $BASE
     FEAT=$(python bestck.py $CK/$FEATDIR)
     python main.py --job_name ${TAG}_joint --training_stage joint --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO $WF --pretrained_weights "$FEAT"
     JCK=$(python bestck.py $CK/${TAG}_joint)
     python main.py --job_name ${TAG}_proj --training_stage projection --epochs 1 --batch_size 32 $BASE $PROTO --pretrained_weights "$JCK"
     python main.py --job_name ${TAG}_clf --training_stage classifier --epochs 100 --batch_size 32 --lr 0.0001 --l2 0.00017 --scheduler_type CosineAnnealingLR --patience 10 $BASE $PROTO --pretrained_weights "$CK/${TAG}_proj/${TAG}_proj_projection.pth"
     echo "DONE ${TAG}" ;;
  *) echo "Unknown experiment '$EXP' (use prep|standard|bce|calib)"; exit 1 ;;
esac
