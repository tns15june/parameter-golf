#!/bin/bash
# Final submission run — 8xH100, competitive config
# Strategy: 10L/512d, 3x MLP, LeakyReLU², int6 QAT, EMA, sliding window eval
# Usage: bash dev/run_final.sh

set -e
cd /workspace/parameter-golf

NUM_UNIQUE_LAYERS=10 \
NUM_RECURRENCES=1 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
TIE_EMBEDDINGS=1 \
QAT_BITS=6 \
QAT_START_FRAC=0.15 \
EXPORT_BITS=6 \
EMBED_EXPORT_BITS=8 \
EMA_DECAY=0.9995 \
EVAL_SEQ_LEN=1024 \
EVAL_STRIDE=256 \
COMPRESS_METHOD=lzma \
ROPE_BASE=10000 \
LOGIT_SOFTCAP=30.0 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
RUN_ID=submission_v4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Run complete ==="
echo "Check logs/submission_v3.txt for full log"
echo "Copy train.log to records folder:"
echo "  cp logs/submission_v3.txt records/track_10min_16mb/2026-04_tns15june_v1/train.log"
