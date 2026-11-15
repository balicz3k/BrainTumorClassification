#!/bin/bash
# Walidacja krzyżowa (5 foldów) trzech modeli z domyślnymi hiperparametrami.
# Uruchomienie w tle: nohup scripts/run_cv.sh > out/logs/cv.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
mkdir -p out/logs
for m in cnn efficientnet_b0 resnet50; do
    venv/bin/python -m src.train cv --model $m > out/logs/cv_$m.log 2>&1
done
