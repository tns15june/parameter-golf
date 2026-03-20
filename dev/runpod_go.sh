#!/bin/bash
# One-command RunPod launcher — survives SSH disconnects via nohup
# Usage: bash dev/runpod_go.sh [final|wide|validate]
#   final    — 8xH100 submission run with dim=768 (default)
#   wide     — 8xH100 submission run with dim=1024
#   validate — 1xGPU validation of experiments 4+5 (RoPE + TTT)

set -e
MODE="${1:-final}"
NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

echo "============================================================"
echo "  PARAMETER GOLF — RunPod Launcher"
echo "  Mode: $MODE | GPUs: $NGPUS"
echo "  $(date)"
echo "============================================================"

# Setup
export HF_HOME=/workspace/.cache/huggingface
cd /workspace/parameter-golf
pip install -q -r requirements.txt

# Download dataset if needed
TRAIN_SHARD_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$TRAIN_SHARD_COUNT" -lt 80 ]; then
    echo "Downloading dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi
echo "Dataset ready: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l) train shards"

# Select config
case "$MODE" in
    final)
        echo "Running FINAL submission (dim=768, 8xH100)..."
        bash dev/run_final.sh
        ;;
    wide)
        echo "Running WIDE submission (dim=1024, 8xH100)..."
        bash dev/run_wide.sh
        ;;
    validate)
        echo "Running validation experiments 4+5 (1xGPU)..."
        SHARED="NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 NUM_LAYERS=12 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 MLP_MULT=2 VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50"

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
        echo "Usage: bash dev/runpod_go.sh [final|wide|validate]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Logs: /workspace/parameter-golf/logs/"
echo "============================================================"
