#!/bin/bash
#SBATCH -J evalall
#SBATCH -c4 --mem=16G -G1
#SBATCH -p general
#SBATCH -o evalall_%j.out
source ~/ecg_venv/bin/activate
cd ~/protoecgnet/src
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=~/protoecgnet/experiments/eval_runs/$STAMP
mkdir -p "$OUT"; echo "Logs -> $OUT"
EVALS=(
  calibration_eval.py calibration_eval_s7.py calibration_eval_s123.py
  calibration_eval_chapman.py calibration_eval_chapman_s7.py calibration_eval_chapman_s123.py
  calibration_eval_georgia.py calibration_eval_georgia_s7.py calibration_eval_georgia_s123.py
)
for e in "${EVALS[@]}"; do
  echo ""; echo "======== $e ========"
  python "$e" > "$OUT/${e%.py}.log" 2>&1 && echo "OK   $e" || echo "FAIL $e (see log)"
done
echo ""; echo "ALL DONE -> $OUT"