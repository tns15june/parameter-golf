#!/bin/bash
# Final submission run — 8xH100, full config
# Usage: bash dev/run_final.sh

set -e
cd /workspace/parameter-golf

NUM_UNIQUE_LAYERS=3 \
NUM_RECURRENCES=4 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=6 \
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
RUN_ID=submission_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Run complete ==="
echo "Check logs/submission_v1.txt for full log"
echo "Copy train.log to records folder:"
echo "  cp logs/submission_v1.txt records/track_10min_16mb/2026-04_tns15june_v1/train.log"
