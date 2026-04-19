#!/bin/bash
# Per-component smoke tests for the frontier port on 1xH100.
# Small SP1024 config (dim=128, 4L, 500 iters) — exercises new code paths
# cheaply. SP8192 data prep is expensive (~2hr retokenize), so smokes use
# SP1024 and the real 8xH100 run uses SP8192.
# Usage:
#   bash dev/smoke_frontier.sh            # run all smokes sequentially
#   bash dev/smoke_frontier.sh qk         # one smoke
#   bash dev/smoke_frontier.sh full       # only the full-stack smoke

set -e
cd /workspace/parameter-golf

if [ -z "${HF_TOKEN:-}" ] && [ -f huggingface_token.txt ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < huggingface_token.txt)"
fi

pip install -q -r requirements.txt

COMPONENT="${1:-all}"
DATA_DIR="data/datasets/fineweb10B_sp1024"

# Only download SP1024 if the requested component needs it. `sp8192` exercises a
# separate data path and any other SP1024-requiring case triggers setup.
if [ "$COMPONENT" != "sp8192" ]; then
    if [ ! -d "$DATA_DIR" ] || [ $(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l) -lt 1 ]; then
        echo "Downloading SP1024 data (4 shards for smoke)..."
        python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4
    fi
fi

SMOKE_COMMON="DATA_PATH=$DATA_DIR TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 \
NUM_UNIQUE_LAYERS=4 NUM_RECURRENCES=1 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
TRAIN_SEQ_LEN=512 TIE_EMBEDDINGS=1 ITERATIONS=500 TRAIN_BATCH_TOKENS=16384 \
MAX_WALLCLOCK_SECONDS=300 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=50 \
WARMUP_STEPS=5 WARMDOWN_ITERS=50 LOGIT_SOFTCAP=30.0 ROPE_BASE=10000"

run_smoke() {
    local name="$1"; shift
    echo ""
    echo "===== SMOKE: $name ====="
    env $SMOKE_COMMON RUN_ID="smoke_$name" "$@" torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -15
    local status=${PIPESTATUS[0]}
    if [ $status -eq 0 ]; then
        echo "$name: PASS"
    else
        echo "$name: FAIL (exit $status)"
        exit 1
    fi
}

ALL=false
[ "$COMPONENT" = "all" ] && ALL=true

$ALL || [ "$COMPONENT" = "baseline" ] && run_smoke baseline
$ALL || [ "$COMPONENT" = "qk" ] && run_smoke qk QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "pr" ] && run_smoke pr PARALLEL_RESIDUALS=1 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "plres" ] && run_smoke plres PARALLEL_LATER_RESIDUALS=1 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "rec" ] && run_smoke rec TARGETED_RECURRENCE=1 NUM_RECURRENCES=2 RECURRENCE_START_LAYER=1 RECURRENCE_END_LAYER=2 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "norm" ] && run_smoke norm LAYERWISE_NORM_SCALE=1 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "muonrow" ] && run_smoke muonrow MUON_ROW_NORM=1 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "prope" ] && run_smoke prope ROPE_FRACTION=0.25 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "gptq" ] && run_smoke gptq QUANT_METHOD=gptq GPTQ_EMBED=0 USE_SDCLIP=1 SDCLIP_K=2.5 EXPORT_BITS=6 EMA_DECAY=0.9965 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "brotli" ] && run_smoke brotli COMPRESS_METHOD=brotli BYTE_SHUFFLE_STRIDE=2 EXPORT_BITS=6 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "ttt" ] && run_smoke ttt TTT_ENABLED=1 TTT_CHUNK_TOKENS=1024 TTT_EPOCHS=2 TTT_MAX_CHUNKS=4 EVAL_STRIDE=256 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "ttt_adapt" ] && run_smoke ttt_adapt TTT_ADAPT_ENABLED=1 TTT_ADAPT_EVERY=8 TTT_ADAPT_START_FRAC=0.1 TTT_ADAPT_LAMBDA=0.1 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "ctrl_reg" ] && run_smoke ctrl_reg CTRL_SURFACE_LAMBDA=0.1 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "wd" ] && run_smoke wd MUON_WEIGHT_DECAY=0.09 QK_GAIN_INIT=5.25
$ALL || [ "$COMPONENT" = "full" ] && run_smoke full PARALLEL_RESIDUALS=1 PARALLEL_LATER_RESIDUALS=1 TARGETED_RECURRENCE=1 NUM_RECURRENCES=2 RECURRENCE_START_LAYER=1 RECURRENCE_END_LAYER=2 LAYERWISE_NORM_SCALE=1 MUON_ROW_NORM=1 ROPE_FRACTION=0.25 QUANT_METHOD=gptq GPTQ_EMBED=0 USE_SDCLIP=1 SDCLIP_K=2.5 EXPORT_BITS=6 EMA_DECAY=0.9965 TTT_ENABLED=1 TTT_CHUNK_TOKENS=1024 TTT_EPOCHS=2 TTT_MAX_CHUNKS=4 EVAL_STRIDE=256 TTT_ADAPT_ENABLED=1 TTT_ADAPT_EVERY=8 TTT_ADAPT_START_FRAC=0.1 TTT_ADAPT_LAMBDA=0.1 CTRL_SURFACE_LAMBDA=0.1 MUON_WEIGHT_DECAY=0.09 QK_GAIN_INIT=5.25 COMPRESS_METHOD=brotli BYTE_SHUFFLE_STRIDE=2

# SP8192 micro-smoke: exercises the real tokenizer / shard / vocab code path.
# Skipped when SP8192 data hasn't been produced yet (Phase A not run).
SP8192_DATA_DIR="data/datasets/fineweb10B_sp8192"
SP8192_TOK="data/tokenizers/fineweb_8192_bpe.model"
if { $ALL || [ "$COMPONENT" = "sp8192" ]; }; then
    if [ -d "$SP8192_DATA_DIR" ] && [ -f "$SP8192_TOK" ] && [ $(ls "$SP8192_DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l) -ge 1 ]; then
        env DATA_PATH="$SP8192_DATA_DIR" TOKENIZER_PATH="$SP8192_TOK" VOCAB_SIZE=8192 \
            NUM_UNIQUE_LAYERS=4 NUM_RECURRENCES=1 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 \
            TRAIN_SEQ_LEN=512 TIE_EMBEDDINGS=1 ITERATIONS=300 TRAIN_BATCH_TOKENS=16384 \
            MAX_WALLCLOCK_SECONDS=180 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=50 \
            WARMUP_STEPS=5 WARMDOWN_ITERS=30 LOGIT_SOFTCAP=30.0 ROPE_BASE=10000 \
            RUN_ID=smoke_sp8192 \
            torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -15
        [ ${PIPESTATUS[0]} -eq 0 ] && echo "sp8192: PASS" || { echo "sp8192: FAIL"; exit 1; }
    else
        echo "sp8192: SKIPPED (run Phase A data prep: bash dev/runpod_go.sh frontier would trigger it, or invoke data/download_hf_docs_and_tokenize.py directly)"
    fi
fi

echo ""
echo "=== smoke_frontier.sh complete ==="
