#!/bin/bash
#SBATCH -J geo_smoke
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o %x_%j.out
set -e
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
CK=$HOME/protoecgnet/experiments/checkpoints
BASE="--dimension 1D --backbone resnet1d18 --sampling_rate 100 --label_set georgia --custom_groups True --dataset georgia --dropout 0.35 --num_workers 4 --save_weights True --checkpoint_dir $CK --log_dir $HOME/protoecgnet/experiments/logs --test_dir $HOME/protoecgnet/experiments/test_results --seed 42"
python main.py --job_name georgia_fix_feat --training_stage feature_extractor --epochs 3 --batch_size 32 --patience 10 $BASE
echo "===== bestck ====="
python bestck.py $CK/georgia_fix_feat
echo "GEORGIA SMOKE DONE"
