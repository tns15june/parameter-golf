#!/bin/bash
# Per-component smoke tests for the frontier port on 1xH100.
# Runs a small SP8192 config (~500 iters, dim=128) with each frontier feature
# toggled on in isolation, then a "full" run with everything enabled.
#
# Usage:
#   bash dev/smoke_frontier.sh            # run all smokes sequentially
#   bash dev/smoke_frontier.sh qk         # only the QK gain smoke
#   bash dev/smoke_frontier.sh full       # only the full-stack smoke
#
# Components: baseline, qk, pr (parallel residuals), rec (targeted recurrence),
#             gptq (GPTQ + SDClip + embed), ttt (legal score-first TTT),
#             wd (Muon weight decay), full

set -e
cd /workspace/parameter-golf

if [ -z "${HF_TOKEN:-}" ] && [ -f huggingface_token.txt ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < huggingface_token.txt)"
fi

pip install -q -r requirements.txt

# Ensure SP1024 data (SP8192 would need local retokenization from docs_selected.jsonl;
# not feasible on this branch's budget). 4 shards are enough for a smoke.
DATA_DIR="data/datasets/fineweb10B_sp1024"
if [ ! -d "$DATA_DIR" ] || [ $(ls "$DATA_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l) -lt 1 ]; then
    echo "Downloading SP1024 data (4 shards for smoke)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4
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
    env $SMOKE_COMMON RUN_ID="smoke_$name" "$@" torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tail -12
    local status=${PIPESTATUS[0]}
    if [ $status -eq 0 ]; then
        echo "$name: PASS"
    else
        echo "$name: FAIL (exit $status)"
        exit 1
    fi
}

COMPONENT="${1:-all}"
ALL=false
[ "$COMPONENT" = "all" ] && ALL=true

if $ALL || [ "$COMPONENT" = "baseline" ]; then
    run_smoke baseline
fi
if $ALL || [ "$COMPONENT" = "qk" ]; then
    run_smoke qk QK_GAIN_INIT=5.25
fi
if $ALL || [ "$COMPONENT" = "pr" ]; then
    run_smoke pr PARALLEL_RESIDUALS=1 QK_GAIN_INIT=5.25
fi
if $ALL || [ "$COMPONENT" = "rec" ]; then
    run_smoke rec TARGETED_RECURRENCE=1 NUM_RECURRENCES=2 RECURRENCE_START_LAYER=1 RECURRENCE_END_LAYER=2 QK_GAIN_INIT=5.25
fi
if $ALL || [ "$COMPONENT" = "gptq" ]; then
    run_smoke gptq QUANT_METHOD=gptq GPTQ_EMBED=1 USE_SDCLIP=1 SDCLIP_K=2.5 QAT_BITS=6 EXPORT_BITS=6 EMA_DECAY=0.9995 QK_GAIN_INIT=5.25
fi
if $ALL || [ "$COMPONENT" = "ttt" ]; then
    run_smoke ttt TTT_ENABLED=1 TTT_CHUNK_TOKENS=1024 TTT_EPOCHS=2 EVAL_STRIDE=256 QK_GAIN_INIT=5.25
fi
if $ALL || [ "$COMPONENT" = "wd" ]; then
    run_smoke wd MUON_WEIGHT_DECAY=0.09 QK_GAIN_INIT=5.25
fi
if $ALL || [ "$COMPONENT" = "full" ]; then
    run_smoke full PARALLEL_RESIDUALS=1 TARGETED_RECURRENCE=1 NUM_RECURRENCES=2 RECURRENCE_START_LAYER=1 RECURRENCE_END_LAYER=2 \
        QUANT_METHOD=gptq GPTQ_EMBED=1 USE_SDCLIP=1 SDCLIP_K=2.5 QAT_BITS=6 EXPORT_BITS=6 EMA_DECAY=0.9995 \
        TTT_ENABLED=1 TTT_CHUNK_TOKENS=1024 TTT_EPOCHS=2 EVAL_STRIDE=256 \
        MUON_WEIGHT_DECAY=0.09 QK_GAIN_INIT=5.25 COMPRESS_METHOD=lzma
fi

echo ""
echo "=== smoke_frontier.sh complete ==="
