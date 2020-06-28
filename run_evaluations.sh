#!/bin/bash

set -e

for filename in "configs/test/tless"/*; do
    python evaluate.py --config "$filename"
done

for filename in "configs/test/baselines/upna"/*; do
    python evaluate.py --config "$filename"
done

for filename in "configs/test/baselines/idiap"/*; do
    python evaluate.py --config "$filename"
done

for filename in "configs/test/calibration/upna"/*; do
    python evaluate.py --config "$filename"
done

for filename in "configs/test/calibration/idiap"/*; do
    python evaluate.py --config "$filename"
done

for filename in "configs/test/tless_multimodal"/*; do
    python evaluate.py --config "$filename"
done 
