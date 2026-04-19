#!/bin/bash
# Frontier submission run — SP8192 stack targeting ~1.08-1.10 BPB
# Usage: bash dev/run_frontier.sh

set -e
cd /workspace/parameter-golf

# SP8192 data + tokenizer
DATA_PATH="${DATA_PATH:-data/datasets/fineweb10B_sp8192}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/fineweb_8192_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-8192}" \
\
NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-11}" \
NUM_RECURRENCES="${NUM_RECURRENCES:-1}" \
MODEL_DIM="${MODEL_DIM:-512}" \
NUM_HEADS="${NUM_HEADS:-8}" \
NUM_KV_HEADS="${NUM_KV_HEADS:-4}" \
MLP_MULT="${MLP_MULT:-2}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}" \
QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}" \
PARALLEL_RESIDUALS="${PARALLEL_RESIDUALS:-1}" \
ROPE_BASE="${ROPE_BASE:-10000}" \
LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}" \
\
QAT_BITS="${QAT_BITS:-6}" \
QAT_START_FRAC="${QAT_START_FRAC:-0.15}" \
EXPORT_BITS="${EXPORT_BITS:-6}" \
EMBED_EXPORT_BITS="${EMBED_EXPORT_BITS:-6}" \
QUANT_METHOD="${QUANT_METHOD:-gptq}" \
GPTQ_EMBED="${GPTQ_EMBED:-1}" \
USE_SDCLIP="${USE_SDCLIP:-1}" \
SDCLIP_K="${SDCLIP_K:-2.5}" \
\
EMA_DECAY="${EMA_DECAY:-0.9995}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}" \
EVAL_STRIDE="${EVAL_STRIDE:-256}" \
COMPRESS_METHOD="${COMPRESS_METHOD:-lzma}" \
\
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}" \
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}" \
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}" \
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}" \
RUN_ID="${RUN_ID:-submission_frontier}" \
SEED="${SEED:-1337}" \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Frontier run complete ==="
LOG_PATH="logs/${RUN_ID:-submission_frontier}.txt"
if [ -f "$LOG_PATH" ]; then
    echo "Log at $LOG_PATH (not auto-filling submission.json yet — fill after all seeds complete)"
else
    echo "WARN: expected log at $LOG_PATH not found."
fi
