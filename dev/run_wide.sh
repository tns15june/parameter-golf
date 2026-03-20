#!/bin/bash
# Wider model submission run — 8xH100, dim=1024
# 23M params → ~11.3MB compressed (4.7MB headroom under 16MB)
# Usage: bash dev/run_wide.sh

set -e
cd /workspace/parameter-golf

NUM_UNIQUE_LAYERS=3 \
NUM_RECURRENCES=4 \
MODEL_DIM=1024 \
NUM_HEADS=16 \
NUM_KV_HEADS=8 \
MLP_MULT=2 \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
TIE_EMBEDDINGS=1 \
QAT_BITS=4 \
QAT_START_FRAC=0.25 \
EXPORT_BITS=4 \
EMBED_EXPORT_BITS=8 \
EVAL_SEQ_LEN=4096 \
TTT_ENABLED=1 \
TTT_LR=1e-5 \
ROPE_BASE=10000 \
LOGIT_SOFTCAP=30.0 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
RUN_ID=submission_wide_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Run complete ==="
echo "Check logs/submission_wide_v1.txt for full log"
