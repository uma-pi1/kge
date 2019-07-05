#!/bin/bash
#
# Run tests (very basic right now):
# - Train example models for 1 epoch
set -e

for model in complex rt3 srt3 conve ; do
    python kge.py start examples/toy-$model-train.yaml --folder `mktemp -du` --train.max_epochs 1 --valid.every 1 --job.device cpu
done

echo ALL TESTS SUCCEEDED
