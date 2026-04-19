#!/bin/bash
# Final submission run with Self-Generated GPTQ — 8xH100, ~10 min training.
# Use AFTER dev/run_gptq_ablate.sh confirms GPTQ improves over amax.
#
# Defaults to int6 GPTQ (safest). Override EXPORT_BITS=4 for the wider-model config
# if ablation showed int4 gap < 0.015 BPB.
#
# Usage:
#   bash dev/run_final_gptq.sh              # int6 GPTQ, same arch as run_final.sh
#   bash dev/run_final_gptq.sh wide         # int4 GPTQ + dim=768 (if ablation says so)

set -e
cd /workspace/parameter-golf

MODE="${1:-int6}"

case "$MODE" in
    int6)
        # Conservative: same arch as current baseline but GPTQ instead of amax.
        NUM_UNIQUE_LAYERS=10 NUM_RECURRENCES=1 \
        MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
        VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 \
        QAT_BITS=0 \
        EXPORT_BITS=6 EMBED_EXPORT_BITS=8 \
        QUANT_METHOD=gptq GPTQ_CALIB_TOKENS=65536 GPTQ_DAMP_PERCENT=0.01 \
        EMA_DECAY=0.9995 EVAL_SEQ_LEN=2048 EVAL_STRIDE=256 \
        COMPRESS_METHOD=lzma \
        ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 \
        TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=600 \
        VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
        RUN_ID=submission_gptq_int6 \
        torchrun --standalone --nproc_per_node=8 train_gpt.py
        ;;
    wide)
        # Aggressive: int4 GPTQ + dim=768 (uses the extra byte budget for model capacity).
        # Only run if ablation confirmed gptq_int4 post_bpb < gptq_int6 post_bpb.
        NUM_UNIQUE_LAYERS=10 NUM_RECURRENCES=1 \
        MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 MLP_MULT=3 \
        VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 \
        QAT_BITS=0 \
        EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
        QUANT_METHOD=gptq GPTQ_CALIB_TOKENS=65536 GPTQ_DAMP_PERCENT=0.01 \
        EMA_DECAY=0.9995 EVAL_SEQ_LEN=2048 EVAL_STRIDE=256 \
        COMPRESS_METHOD=lzma \
        ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 \
        TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=600 \
        VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
        RUN_ID=submission_gptq_wide \
        torchrun --standalone --nproc_per_node=8 train_gpt.py
        ;;
    *)
        echo "Usage: bash dev/run_final_gptq.sh [int6|wide]"
        exit 1
        ;;
esac

echo ""
echo "=== Run complete ==="
echo "Next: python3 dev/fill_submission.py logs/submission_gptq_${MODE}.txt"
echo "Then: bash dev/sync_submission.sh"
