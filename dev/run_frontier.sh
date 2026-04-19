#!/bin/bash
# Full SP8192 frontier submission run (8xH100).
# Components: SP8192 data/tokenizer, 11L x 512d x MLP4x, targeted recurrence,
# parallel residuals (same-block + from-later-layers via softmax mix), partial
# RoPE 25%, layerwise RMSNorm scale, QK gain 5.25, MuonEq-R + WD 0.09, EMA
# 0.9965, GPTQ + SDClip (no QAT), int8 embeddings, byte-shuffle + Brotli
# packing, legal score-first sliding TTT, first-order TTT-adaptable training,
# control-surface regularizer.
#
# Usage: bash dev/run_frontier.sh

set -e
cd /workspace/parameter-golf

DATA_PATH="${DATA_PATH:-data/datasets/fineweb10B_sp8192}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/fineweb_8192_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-8192}" \
NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-11}" \
NUM_RECURRENCES="${NUM_RECURRENCES:-3}" \
TARGETED_RECURRENCE="${TARGETED_RECURRENCE:-1}" \
RECURRENCE_START_LAYER="${RECURRENCE_START_LAYER:-4}" \
RECURRENCE_END_LAYER="${RECURRENCE_END_LAYER:-7}" \
MODEL_DIM="${MODEL_DIM:-512}" \
NUM_HEADS="${NUM_HEADS:-8}" \
NUM_KV_HEADS="${NUM_KV_HEADS:-4}" \
MLP_MULT="${MLP_MULT:-4}" \
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}" \
TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}" \
QK_GAIN_INIT="${QK_GAIN_INIT:-5.25}" \
ROPE_BASE="${ROPE_BASE:-10000}" \
ROPE_FRACTION="${ROPE_FRACTION:-0.25}" \
PARALLEL_RESIDUALS="${PARALLEL_RESIDUALS:-1}" \
PARALLEL_LATER_RESIDUALS="${PARALLEL_LATER_RESIDUALS:-1}" \
LAYERWISE_NORM_SCALE="${LAYERWISE_NORM_SCALE:-1}" \
LOGIT_SOFTCAP="${LOGIT_SOFTCAP:-30.0}" \
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.09}" \
MUON_ROW_NORM="${MUON_ROW_NORM:-1}" \
EXPORT_BITS="${EXPORT_BITS:-6}" \
EMBED_EXPORT_BITS="${EMBED_EXPORT_BITS:-8}" \
QUANT_METHOD="${QUANT_METHOD:-gptq}" \
GPTQ_EMBED="${GPTQ_EMBED:-0}" \
USE_SDCLIP="${USE_SDCLIP:-1}" \
SDCLIP_K="${SDCLIP_K:-2.5}" \
EMA_DECAY="${EMA_DECAY:-0.9965}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}" \
EVAL_STRIDE="${EVAL_STRIDE:-256}" \
TTT_ENABLED="${TTT_ENABLED:-1}" \
TTT_LR="${TTT_LR:-1e-5}" \
TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-8192}" \
TTT_EPOCHS="${TTT_EPOCHS:-2}" \
TTT_MAX_CHUNKS="${TTT_MAX_CHUNKS:-0}" \
TTT_ADAPT_ENABLED="${TTT_ADAPT_ENABLED:-0}" \
TTT_ADAPT_EVERY="${TTT_ADAPT_EVERY:-32}" \
TTT_ADAPT_START_FRAC="${TTT_ADAPT_START_FRAC:-0.60}" \
TTT_ADAPT_LR="${TTT_ADAPT_LR:-1e-3}" \
TTT_ADAPT_LAMBDA="${TTT_ADAPT_LAMBDA:-0.10}" \
CTRL_SURFACE_LAMBDA="${CTRL_SURFACE_LAMBDA:-0.1}" \
COMPRESS_METHOD="${COMPRESS_METHOD:-brotli}" \
BYTE_SHUFFLE_STRIDE="${BYTE_SHUFFLE_STRIDE:-2}" \
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
    echo "Filling submission.json + copying train.log from $LOG_PATH..."
    python3 dev/fill_submission.py "$LOG_PATH"
else
    echo "WARN: expected log at $LOG_PATH not found."
fi
