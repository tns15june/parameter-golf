## 10L/512d LeakyReLU² + int6 QAT + EMA + Sliding Window

### Approach

Competitive architecture with multiple training and eval-time optimizations:

1. **10-layer 2x MLP architecture**: 10 transformer layers at dimension 512, 8 attention heads with 4 KV heads (GQA), 2x MLP expansion with LeakyReLU² activation, tied embeddings. ~18M params.

2. **Int6 QAT + LZMA export**: Quantization-aware training (6-bit, starting at 15% of training) minimizes the export gap. LZMA compression achieves excellent ratio for int6 values.

3. **EMA (0.9995)**: Exponential moving average of weights during training produces smoother weights that generalize and compress better.

4. **Sliding window eval (seq_len matches train)**: Overlapping windows (stride=256, seq_len=1024) ensure every token has warm context, improving BPB over non-overlapping evaluation. No RoPE extrapolation.

### Configuration

```bash
NUM_UNIQUE_LAYERS=10 NUM_RECURRENCES=1 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=2 VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 \
QAT_BITS=6 QAT_START_FRAC=0.15 EXPORT_BITS=6 EMBED_EXPORT_BITS=8 \
EMA_DECAY=0.9995 EVAL_SEQ_LEN=1024 EVAL_STRIDE=256 \
COMPRESS_METHOD=lzma \
ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Architecture

- **Depth**: 10 layers (no recurrence)
- **Width**: 512
- **MLP**: 2x expansion with LeakyReLU² (slope=0.01)
- **Params**: ~18M total
- **Export**: int6 QAT blocks + int8 embeddings, LZMA compressed
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
