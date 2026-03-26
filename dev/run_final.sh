#!/bin/bash
# Final submission run — 8xH100, full config
# Strategy: baseline arch (9L/512d), no QAT, int6+lzma export, n-gram eval
# Usage: bash dev/run_final.sh

set -e
cd /workspace/parameter-golf

NUM_UNIQUE_LAYERS=9 \
NUM_RECURRENCES=1 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
TIE_EMBEDDINGS=1 \
QAT_BITS=0 \
EXPORT_BITS=6 \
EMBED_EXPORT_BITS=8 \
EVAL_SEQ_LEN=1024 \
NGRAM_ENABLED=1 \
NGRAM_MAX_ORDER=5 \
NGRAM_ALPHA=0.2 \
COMPRESS_METHOD=lzma \
ROPE_BASE=10000 \
LOGIT_SOFTCAP=30.0 \
TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
RUN_ID=submission_v2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Run complete ==="
echo "Check logs/submission_v2.txt for full log"
echo "Copy train.log to records folder:"
echo "  cp logs/submission_v2.txt records/track_10min_16mb/2026-04_tns15june_v1/train.log"
