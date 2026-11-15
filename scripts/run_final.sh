#!/bin/bash
# Etap po walidacji krzyżowej (scripts/run_cv.sh):
#   1. strojenie Optuną wybranej architektury (tylko foldy walidacyjne),
#   2. pełne 5-fold CV z najlepszymi parametrami,
#   3. modele końcowe na całym zbiorze treningowym,
#   4. jednorazowa ewaluacja wszystkich modeli końcowych na teście.
# Uruchomienie w tle: nohup scripts/run_final.sh efficientnet_b0 > out/logs/final.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
M=${1:-efficientnet_b0}
PY=venv/bin/python
mkdir -p out/logs
$PY -m src.tune --model $M --trials 20 > out/logs/tune_$M.log 2>&1
$PY -m src.train cv --model $M --params out/metrics/optuna_${M}_best.json --tag ${M}_tuned > out/logs/cv_${M}_tuned.log 2>&1
for m in cnn efficientnet_b0 resnet50; do
    $PY -m src.train final --model $m > out/logs/final_$m.log 2>&1
done
$PY -m src.train final --model $M --params out/metrics/optuna_${M}_best.json --tag ${M}_tuned > out/logs/final_${M}_tuned.log 2>&1
$PY -m src.evaluate cnn efficientnet_b0 resnet50 ${M}_tuned > out/logs/evaluate.log 2>&1
