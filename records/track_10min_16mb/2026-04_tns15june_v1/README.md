## Baseline 9L/512d + int6 LZMA + N-gram Eval

### Approach

Baseline architecture with optimized export and eval-time scoring:

1. **Baseline architecture (9L/512d)**: 9 transformer layers at dimension 512, 8 attention heads with 4 KV heads (GQA), 2x MLP expansion, tied embeddings. No depth recurrence — maximizes training steps within the 10-minute wallclock budget.

2. **int6 + LZMA export**: 6-bit quantization for block weights (per-row percentile clipping), 8-bit for embeddings. LZMA compression (preset 9 extreme) achieves ~30% smaller artifacts than zlib, keeping total size well under 16MB.

3. **N-gram eval cache**: Online n-gram statistics (up to order 5) blended with transformer predictions at eval time. Zero artifact bytes — the cache is built from validation data during evaluation.

### Configuration

```bash
NUM_UNIQUE_LAYERS=9 NUM_RECURRENCES=1 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=2 VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 \
QAT_BITS=0 EXPORT_BITS=6 EMBED_EXPORT_BITS=8 \
EVAL_SEQ_LEN=1024 NGRAM_ENABLED=1 NGRAM_MAX_ORDER=5 NGRAM_ALPHA=0.2 \
COMPRESS_METHOD=lzma \
ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Architecture

- **Depth**: 9 layers (no recurrence)
- **Width**: 512
- **Params**: ~4.6M total
- **Export**: int6 blocks + int8 embeddings, LZMA compressed
- **U-Net skips**: Applied across encoder/decoder layer halves

### Key Metrics

*(To be filled after 8xH100 run)*

- Pre-quant eval: `val_loss:___ val_bpb:___`
- Post-quant roundtrip: `val_loss:___ val_bpb:___`
- Train time: `___ms`
- Serialized model: `___ bytes`
- Code size: `___ bytes`
- Total submission size: `___ bytes`

### Included Files

- `train_gpt.py` — complete training + eval script
- `train.log` — full training log from 8xH100 run
- `submission.json` — leaderboard metadata
