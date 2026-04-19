#!/bin/bash
# One-command RunPod launcher — survives SSH disconnects via nohup
# Usage: bash dev/runpod_go.sh [final|frontier|smoke|wide|validate]
#   frontier — 8xH100 SP8192 frontier-port submission (the competition run)
#   smoke    — 1xH100 per-component smoke tests (calls dev/smoke_frontier.sh)
#   final    — 8xH100 SP1024 v4 fallback run (beats baseline only)
#   wide     — 8xH100 SP1024 dim=1024 variant
#   validate — 1xGPU legacy validation experiments (SP1024)

set -e
MODE="${1:-frontier}"
NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

# frontier mode uses SP8192 (trained + retokenized locally via tokenizer_specs.json).
# smoke + other modes use SP1024 (pre-published in upstream manifest).
if [ "$MODE" = "frontier" ]; then
    VARIANT=sp8192
else
    VARIANT=sp1024
fi
DATA_DIR="data/datasets/fineweb10B_${VARIANT}"

echo "============================================================"
echo "  PARAMETER GOLF — RunPod Launcher"
echo "  Mode: $MODE | GPUs: $NGPUS"
echo "  $(date)"
echo "============================================================"

# Setup
export HF_HOME=/workspace/.cache/huggingface
cd /workspace/parameter-golf

# Optional: load HF_TOKEN from huggingface_token.txt if present (gitignored).
# Useful to avoid rate-limits on dataset downloads. Never logs the token.
if [ -z "${HF_TOKEN:-}" ] && [ -f huggingface_token.txt ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < huggingface_token.txt)"
    if [ -n "$HF_TOKEN" ]; then
        echo "HF_TOKEN loaded from huggingface_token.txt ($(echo -n "$HF_TOKEN" | wc -c) chars)"
    fi
fi

pip install -q -r requirements.txt

# Download dataset if needed (variant chosen by mode above)
TRAIN_SHARD_COUNT=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
# For smoke mode a small subset is enough; for real runs we need the full 80 shards.
MIN_SHARDS=$([ "$MODE" = "smoke" ] && echo 4 || echo 80)
if [ "$TRAIN_SHARD_COUNT" -lt "$MIN_SHARDS" ]; then
    echo "Downloading $VARIANT dataset (have $TRAIN_SHARD_COUNT shards, need $MIN_SHARDS)..."
    # SP8192 requires --also-download-docs to trigger local tokenizer training + retokenization
    # (manifest only publishes SP1024 pre-tokenized; SP8192 is produced locally from docs_selected.jsonl).
    EXTRA_ARG=""
    [ "$VARIANT" = "sp8192" ] && EXTRA_ARG="--also-download-docs"
    if [ "$MODE" = "smoke" ]; then
        python3 data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards 4 $EXTRA_ARG
    else
        python3 data/cached_challenge_fineweb.py --variant "$VARIANT" $EXTRA_ARG
    fi
fi
echo "Dataset ready: $(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l) train shards ($VARIANT)"

# Select config
case "$MODE" in
    frontier)
        echo "Running FRONTIER submission (SP8192 stack, 8xH100)..."
        bash dev/run_frontier.sh
        ;;
    smoke)
        echo "Running FRONTIER smoke tests (1xH100)..."
        bash dev/smoke_frontier.sh "${2:-all}"
        ;;
    final)
        echo "Running FINAL submission (dim=512, SP1024 v4, 8xH100)..."
        bash dev/run_final.sh
        ;;
    wide)
        echo "Running WIDE submission (dim=1024, 8xH100)..."
        bash dev/run_wide.sh
        ;;
    validate)
        echo "Running validation experiments 4+5 (1xGPU)..."
        SHARED="NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 MLP_MULT=2 VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50"

        echo ""
        echo "=== Experiment 4: RoPE 4x scaling ==="
        env $SHARED QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 EMBED_EXPORT_BITS=8 EVAL_SEQ_LEN=4096 RUN_ID=validate_rope \
            torchrun --standalone --nproc_per_node=1 train_gpt.py || true

        echo ""
        echo "=== Experiment 5: TTT ==="
        env $SHARED QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 EMBED_EXPORT_BITS=8 TTT_ENABLED=1 TTT_LR=1e-5 RUN_ID=validate_ttt \
            torchrun --standalone --nproc_per_node=1 train_gpt.py || true
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash dev/runpod_go.sh [frontier|smoke|final|wide|validate]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Logs: /workspace/parameter-golf/logs/"
echo "============================================================"
