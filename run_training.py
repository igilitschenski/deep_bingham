#!/bin/bash

# Exit on failure
set -e

for filename in "configs/tless"/*; do
    python train.py --config "$filename"
done

for filename in "configs/baselines/upna"/*; do
    python train.py --config "$filename"
done

for filename in "configs/baselines/idiap"/*; do
    python train.py --config "$filename"
done

for filename in "configs/calibration/upna"/*; do
    python train.py --config "$filename"
done

for filename in "configs/calibration/idiap"/*; do
    python train.py --config "$filename"
done

for filename in "configs/tless_multimodal"/*; do
    python train.py --config "$filename"
done 
