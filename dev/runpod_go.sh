#!/bin/bash
# One-command RunPod launcher — survives SSH disconnects via nohup
# Usage: bash dev/runpod_go.sh [MODE]
#   frontier     — 8xH100 SP8192 frontier-port submission (the competition run).
#                  Requires SP8192 data on the volume; runs prep first if missing.
#   prep-sp8192  — DEPRECATED since 2026-04-21: SP8192 is now pulled pre-tokenized
#                  from kevclark/parameter-golf on HF during normal `frontier` mode
#                  (~10 min, ~$1-2). Use this mode only if you explicitly need to
#                  retokenize locally (SP8192_FORCE_LOCAL_TOKENIZE=1, ~$7, 2-3 hr).
#   smoke [case] — 1xH100 per-component smoke tests (calls dev/smoke_frontier.sh).
#                  case=sp8192 exercises the SP8192 code path (requires Phase A
#                  data to exist on the volume). Other cases use SP1024.
#   final        — 8xH100 SP1024 v4 fallback run (beats baseline only)
#   wide         — 8xH100 SP1024 dim=1024 variant
#   validate     — 1xGPU legacy validation experiments (SP1024)

set -e
MODE="${1:-frontier}"
NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

# Variant selection: frontier + prep-sp8192 use SP8192 (retokenized locally).
# smoke + other modes use SP1024 (pre-published). The sp8192 smoke case checks
# its own data independently so we don't need SP1024 here — gate that below.
case "$MODE" in
    frontier|prep-sp8192) VARIANT=sp8192 ;;
    *) VARIANT=sp1024 ;;
esac
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

# Skip dataset setup entirely for `smoke sp8192` — that case gates on its own
# SP8192 data existence and doesn't need SP1024 downloaded.
if [ "$MODE" = "smoke" ] && [ "${2:-all}" = "sp8192" ]; then
    SKIP_DATA_SETUP=1
fi

# Download dataset if needed (variant chosen by mode above).
# 2026-04-21: SP8192 is now available PRE-TOKENIZED from kevclark/parameter-golf
# on HuggingFace (same mirror bigbag uses for 1.0810 reproduction). This is ~15x
# cheaper than local retokenize (~$1 download vs ~$7 on 1xH100 for 2-3 hr).
TRAIN_SHARD_COUNT=$(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
MIN_SHARDS=$([ "$MODE" = "smoke" ] && echo 4 || echo 80)
if [ "${SKIP_DATA_SETUP:-0}" = "0" ] && [ "$TRAIN_SHARD_COUNT" -lt "$MIN_SHARDS" ]; then
    if [ "$VARIANT" = "sp8192" ]; then
        # Prefer the pre-tokenized HF mirror. Fall back to local retokenize only if
        # SP8192_FORCE_LOCAL_TOKENIZE=1 is set (e.g. to reproduce the legacy path).
        if [ "${SP8192_FORCE_LOCAL_TOKENIZE:-0}" = "1" ]; then
            echo "Generating SP8192 data locally via download_hf_docs_and_tokenize.py (forced)..."
            python3 data/download_hf_docs_and_tokenize.py \
                --output-root data \
                --tokenizer-config data/tokenizer_specs_sp8192.json \
                --skip-byte
        else
            echo "Downloading pre-tokenized SP8192 from kevclark/parameter-golf on HF..."
            if [ "$MODE" = "smoke" ]; then
                MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
                    python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 4
            else
                MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
                    python3 data/cached_challenge_fineweb.py --variant sp8192
            fi
        fi
    else
        echo "Downloading $VARIANT dataset (have $TRAIN_SHARD_COUNT shards, need $MIN_SHARDS)..."
        if [ "$MODE" = "smoke" ]; then
            python3 data/cached_challenge_fineweb.py --variant "$VARIANT" --train-shards 4
        else
            python3 data/cached_challenge_fineweb.py --variant "$VARIANT"
        fi
    fi
fi
if [ "${SKIP_DATA_SETUP:-0}" = "0" ]; then
    echo "Dataset ready: $(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l) train shards ($VARIANT)"
fi

# Select config
case "$MODE" in
    frontier)
        if [ "$NGPUS" -lt 8 ]; then
            echo "ERROR: frontier mode runs torchrun --nproc_per_node=8 but this pod has $NGPUS GPU(s)."
            echo "  - For data prep only on 1xH100, use: bash dev/runpod_go.sh prep-sp8192"
            echo "  - For the real run, provision an 8xH100 pod first."
            exit 1
        fi
        echo "Running FRONTIER submission (SP8192 stack, 8xH100)..."
        bash dev/run_frontier.sh
        ;;
    prep-sp8192)
        echo "SP8192 data prep complete. Skipping training (mode=prep-sp8192)."
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
        echo "Usage: bash dev/runpod_go.sh [frontier|prep-sp8192|smoke|final|wide|validate]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "  Logs: /workspace/parameter-golf/logs/"
echo "============================================================"
