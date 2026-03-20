#!/bin/bash
# RunPod setup script — run this after SSH into the pod
# Usage: bash runpod_setup.sh

set -e

echo "=== Setting up Parameter Golf ==="

cd /workspace

# Clone from your fork
if [ ! -d "parameter-golf" ]; then
    git clone --branch submission-v1 https://github.com/tns15june/parameter-golf.git
    cd parameter-golf
else
    cd parameter-golf
    git checkout submission-v1
    git pull --ff-only
fi

# Download dataset
echo "=== Downloading dataset ==="
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "=== Setup complete ==="
echo ""
echo "To run baseline (sanity check):"
echo "  torchrun --standalone --nproc_per_node=1 train_gpt.py"
echo ""
echo "To run depth recurrence test (1 GPU, quick):"
echo "  NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 \\"
echo "  torchrun --standalone --nproc_per_node=1 train_gpt.py"
echo ""
echo "To run full submission config (8 GPUs, final):"
echo "  bash dev/run_final.sh"
