#!/bin/bash
# Frontier-features submission run (8xH100).
#
# IMPORTANT: This is SP1024 with frontier features, NOT the SP8192 leaderboard
# stack. SP8192 is not published in the upstream willdepueoai/parameter-golf
# manifest, and retokenizing from docs_selected.jsonl would add hours + $ that
# this branch can't afford.
#
# Features enabled vs submission-v1 baseline:
#   QK_GAIN_INIT=5.25, PARALLEL_RESIDUALS=1, TARGETED_RECURRENCE=1 (11L, mid 4..7 x3),
#   QUANT_METHOD=gptq (no embed, no SDClip — both caused >2 BPB export gap on smoke),
#   QAT_BITS=6, EXPORT_BITS=6, EMBED_EXPORT_BITS=8 (embeds at int8 is safer),
#   EMA_DECAY=0.9995, EVAL_STRIDE=256 (sliding eval at train seq_len, no RoPE scale),
#   MUON_WEIGHT_DECAY=0.09, COMPRESS_METHOD=lzma.
#
# TTT_ENABLED=0: our legal score-first sliding TTT implementation iterates
# ~15k chunks of the full 60M-token val, risks the 10-min eval cap. Re-enable
# for a second run once TTT_MAX_CHUNKS can cap safely while still scoring all tokens.
#
# Usage: bash dev/run_frontier.sh

set -e
cd /workspace/parameter-golf

# All env-var assignments are on consecutive backslash-continuation lines.
# Do NOT insert comment lines between them — bash terminates the command at a
# comment and silently drops later assignments.
DATA_PATH="${DATA_PATH:-data/datasets/fineweb10B_sp1024}" \
TOKENIZER_PATH="${TOKENIZER_PATH:-data/tokenizers/fineweb_1024_bpe.model}" \
VOCAB_SIZE="${VOCAB_SIZE:-1024}" \
NUM_UNIQUE_LAYERS="${NUM_UNIQUE_LAYERS:-11}" \
NUM_RECURRENCES="${NUM_RECURRENCES:-3}" \
TARGETED_RECURRENCE="${TARGETED_RECURRENCE:-1}" \
RECURRENCE_START_LAYER="${RECURRENCE_START_LAYER:-4}" \
RECURRENCE_END_LAYER="${RECURRENCE_END_LAYER:-7}" \
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
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.09}" \
QAT_BITS="${QAT_BITS:-6}" \
QAT_START_FRAC="${QAT_START_FRAC:-0.15}" \
EXPORT_BITS="${EXPORT_BITS:-6}" \
EMBED_EXPORT_BITS="${EMBED_EXPORT_BITS:-8}" \
QUANT_METHOD="${QUANT_METHOD:-gptq}" \
GPTQ_EMBED="${GPTQ_EMBED:-0}" \
USE_SDCLIP="${USE_SDCLIP:-0}" \
SDCLIP_K="${SDCLIP_K:-2.5}" \
EMA_DECAY="${EMA_DECAY:-0.9995}" \
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-1024}" \
EVAL_STRIDE="${EVAL_STRIDE:-256}" \
TTT_ENABLED="${TTT_ENABLED:-0}" \
TTT_LR="${TTT_LR:-1e-5}" \
TTT_CHUNK_TOKENS="${TTT_CHUNK_TOKENS:-4096}" \
TTT_EPOCHS="${TTT_EPOCHS:-3}" \
TTT_MAX_CHUNKS="${TTT_MAX_CHUNKS:-0}" \
COMPRESS_METHOD="${COMPRESS_METHOD:-lzma}" \
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
    echo "WARN: expected log at $LOG_PATH not found. Run fill_submission.py manually after all seeds are done."
fi
