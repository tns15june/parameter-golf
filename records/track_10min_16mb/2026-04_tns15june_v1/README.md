## Depth Recurrence + int4 QAT + Eval-Time Optimization

### Approach

Three orthogonal techniques stacked together:

1. **Depth Recurrence**: 3 unique transformer blocks shared across 4 recurrences = 12 effective layers. Per-effective-layer control scalars (attn_scales, mlp_scales, resid_mixes, q_gains) allow each recurrence to behave differently while sharing heavy weights (Q, K, V, projection, MLP matrices). This frees parameter budget for a wider model.

2. **int4 Quantization-Aware Training (QAT)**: Fake quantization injected into CastedLinear forward passes after 25% of training. STE (Straight-Through Estimator) lets gradients flow through. int4 export stores values in int8 tensors with restricted [-8,7] range — zlib compresses ~2x better, doubling effective parameter budget within the 16MB limit.

3. **Eval-Time Optimization**: NTK-aware RoPE scaling for 4x longer eval context (4096 vs 1024 training). Optional test-time training adapts embeddings and control scalars on validation data.

### Configuration

```bash
NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 \
MLP_MULT=2 VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1 \
QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 \
EVAL_SEQ_LEN=4096 TTT_ENABLED=1 TTT_LR=1e-5 \
ROPE_BASE=10000 LOGIT_SOFTCAP=30.0 TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Architecture

- **Effective depth**: 12 layers (3 unique x 4 recurrences)
- **Width**: 768 (vs baseline 512)
- **Params**: ~13.2M total
- **Compressed size**: ~7.9MB int4+zlib (well under 16MB)
- **U-Net skips**: Applied across effective layer positions

### Key Metrics

*(To be filled after 8xH100 run)*

- Pre-quant eval: `val_loss:___ val_bpb:___`
- Post-quant roundtrip: `val_loss:___ val_bpb:___`
- Train time: `___ms`
- Serialized model int4+zlib: `___ bytes`
- Code size: `___ bytes`
- Total submission size: `___ bytes`

### Included Files

- `train_gpt.py` — complete training + eval script
- `train.log` — full training log from 8xH100 run
- `submission.json` — leaderboard metadata
