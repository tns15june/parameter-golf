#!/bin/bash
# Quick experiment configs for 1xH100 (~3 min each)
# Usage: bash dev/run_experiments.sh <config_name>
# Configs: baseline, A, B, C, D, qat8, qat4

set -e
cd /workspace/parameter-golf

COMMON="VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 MLP_MULT=2 \
TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=180 VAL_LOSS_EVERY=500 TRAIN_LOG_EVERY=100"

case "${1:-baseline}" in
  baseline)
    echo "=== Baseline: 9x512, no recurrence ==="
    eval "$COMMON NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
    RUN_ID=exp_baseline \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  A)
    echo "=== Config A: 3 unique x 3 rec = 9 eff, dim=640 ==="
    eval "$COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 \
    RUN_ID=exp_A_3x3_640 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  B)
    echo "=== Config B: 3 unique x 4 rec = 12 eff, dim=640 ==="
    eval "$COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 \
    RUN_ID=exp_B_3x4_640 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  C)
    echo "=== Config C: 4 unique x 3 rec = 12 eff, dim=576 ==="
    eval "$COMMON NUM_UNIQUE_LAYERS=4 NUM_RECURRENCES=3 MODEL_DIM=576 NUM_HEADS=8 NUM_KV_HEADS=4 \
    RUN_ID=exp_C_4x3_576 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  D)
    echo "=== Config D: 3 unique x 4 rec = 12 eff, dim=512 ==="
    eval "$COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
    RUN_ID=exp_D_3x4_512 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  qat8)
    echo "=== Best recurrence + int8 QAT ==="
    eval "$COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 \
    QAT_BITS=8 QAT_START_FRAC=0.3 \
    RUN_ID=exp_qat8_3x4_640 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  qat4)
    echo "=== int4 QAT + wider model ==="
    eval "$COMMON NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 \
    QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 \
    RUN_ID=exp_qat4_3x4_768 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py"
    ;;
  *)
    echo "Unknown config: $1"
    echo "Usage: bash dev/run_experiments.sh {baseline|A|B|C|D|qat8|qat4}"
    exit 1
    ;;
esac
