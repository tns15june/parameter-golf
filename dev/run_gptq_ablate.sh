#!/bin/bash
# GPTQ ablation on 1xH100 — ~30 min total, ~$2-3 on RunPod.
# Compares: amax int6 (baseline) vs GPTQ int6 vs GPTQ int4 vs GPTQ int4 + wider model.
# Goal: confirm GPTQ shrinks export_gap enough to justify int4 (which halves weight bytes).
#
# Usage: bash dev/run_gptq_ablate.sh
set -e
cd /workspace/parameter-golf

RESULTS=/workspace/parameter-golf/gptq_ablation.txt
echo "GPTQ ABLATION — $(date)" > "$RESULTS"
echo "GPUs: $(nvidia-smi -L | wc -l)" >> "$RESULTS"

# Shared config — same arch as current best (run_final.sh) but shorter wallclock
# so all four variants complete in ~30 min on 1xH100.
COMMON="NUM_UNIQUE_LAYERS=10 NUM_RECURRENCES=1 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3 VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 \
EMA_DECAY=0.9995 EVAL_SEQ_LEN=2048 EVAL_STRIDE=256 COMPRESS_METHOD=lzma \
ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=360 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=100"

run_one() {
    local name="$1"; shift
    echo ""
    echo "============================================================"
    echo "  $name — $(date +%H:%M:%S)"
    echo "============================================================"
    local t0=$(date +%s)
    set +e
    env $COMMON "$@" RUN_ID="abl_${name}" \
        torchrun --standalone --nproc_per_node=1 train_gpt.py \
        2>&1 | tee "/tmp/abl_${name}.log"
    local rc=${PIPESTATUS[0]}
    set -e
    local dt=$(( $(date +%s) - t0 ))
    local pre=$(grep -oP 'pre_export_bpb:\K[\d.]+' "/tmp/abl_${name}.log" | tail -1)
    local post=$(grep -oP 'post_export_bpb:\K[\d.]+' "/tmp/abl_${name}.log" | tail -1)
    local gap=$(grep -oP 'export_gap:\K[-\d.]+' "/tmp/abl_${name}.log" | tail -1)
    local cbytes=$(grep -oP 'Serialized model \w+: \K\d+' "/tmp/abl_${name}.log" | tail -1)
    echo "$name | rc=$rc | pre=$pre post=$post gap=$gap compressed=$cbytes time=${dt}s" \
        | tee -a "$RESULTS"
}

# 1. Baseline: current winner — amax int6 + int8 embed + LZMA (NO QAT, so we isolate export quality)
run_one "amax_int6_noqat" \
    QAT_BITS=0 \
    EXPORT_BITS=6 EMBED_EXPORT_BITS=8 \
    QUANT_METHOD=amax

# 2. GPTQ int6: same bits, different algorithm — should shrink gap vs (1)
run_one "gptq_int6" \
    QAT_BITS=0 \
    EXPORT_BITS=6 EMBED_EXPORT_BITS=8 \
    QUANT_METHOD=gptq GPTQ_CALIB_TOKENS=16384

# 3. GPTQ int4: half the weight bytes vs int6 — gap must be < 0.015 BPB to be useful
run_one "gptq_int4" \
    QAT_BITS=0 \
    EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
    QUANT_METHOD=gptq GPTQ_CALIB_TOKENS=16384

# 4. GPTQ int4 + wider model (dim=768): uses the int4 savings for more capacity
run_one "gptq_int4_dim768" \
    MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 \
    QAT_BITS=0 \
    EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
    QUANT_METHOD=gptq GPTQ_CALIB_TOKENS=16384

echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
cat "$RESULTS"
echo ""
echo "DECISION GUIDE:"
echo "  If (2).gap < (1).gap by >0.005:  GPTQ works. Use int6 + GPTQ."
echo "  If (3).gap < 0.015:              int4 viable. Prefer wider arch."
echo "  If (4).post_bpb < (3).post_bpb:  dim=768 wins. Use for final 8xH100 run."
echo "  Else:                            stick with current winner."
