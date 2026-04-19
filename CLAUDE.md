# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

**OpenAI Parameter Golf**: Train the best language model (lowest BPB) that fits in a 16MB artifact and trains in ≤10 min on 8×H100 SXM GPUs. Evaluated on FineWeb validation set (first 50K documents), tokenizer-agnostic bits-per-byte.

**Deadline**: April 30, 2026. **Repo**: https://github.com/openai/parameter-golf

### Key Constraints

- Artifact = code bytes + compressed model bytes ≤ 16,000,000 bytes (decimal, not MiB)
- Training: ≤10 min on 8×H100. Evaluation: ≤10 min (separate)
- No network calls or training data access during eval (unless paid for in 16MB)
- New SOTA must beat existing by ≥0.005 nats at p < 0.01
- Tokenizer changes get extra scrutiny; bugs can unjustly improve BPB

### Scoring

```
BPB = (cross_entropy_loss / ln(2)) × (tokens_in_sequence / bytes_in_sequence)
```

### Current Leaderboard (as of 2026-04-04)

| Entry | BPB | Track |
|---|---|---|
| #1 Self-Generated GPTQ + XSA | 1.1147 | 10min/16MB |
| #2 LeakyReLU² + TTT + Parallel Muon | 1.1194 | 10min/16MB |
| #3 EMA + GPTQ-lite + Warmdown | 1.1228 | 10min/16MB |
| Naive Baseline | 1.2244 | 10min/16MB |

Quantization gap from 4-hour run: 0.0325 BPB (pre-quant 1.1749 → post-quant 1.2074).

## Commands

### Local Setup (Mac with Apple Silicon)

```bash
cd parameter-golf
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1  # smoke test
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### RunPod Setup (CUDA)

```bash
# Launch pod from template: https://console.runpod.io/deploy?template=y5cejece4j
cd /workspace && git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train baseline (1×H100):
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Train on 8×H100 (final submission):
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All model/training params are set via env vars. Logs go to `logs/<RUN_ID>.txt`.

### Dataset Download

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024              # full (80 shards, 8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1  # minimal
```

Populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.

## Architecture (train_gpt.py)

Two scripts: `train_gpt.py` (CUDA/PyTorch, the submission script) and `train_gpt_mlx.py` (MLX, for local Apple Silicon iteration). Both must stay under 1500 lines.

### Model (GPT class)

Transformer with these non-standard elements:
- **Depth recurrence**: `num_unique_layers` blocks shared via round-robin across `num_effective_layers` (= unique × recurrences). Per-effective-layer control scalars (attn_scales, mlp_scales, resid_mixes, q_gains) are NOT shared.
- **U-Net skip connections**: Encoder half stores activations; decoder half adds `skip_weight * stored_activation` from mirror layer.
- **Residual mixing**: Each effective layer blends current hidden state with original post-embedding state x0 via learned `resid_mix[0]*x + resid_mix[1]*x0`.
- **CastedLinear**: Weights in fp32, cast to bf16 at compute time. Supports QAT via `_qat_bits` class variable.
- **GQA**: Grouped-query attention (default 8 query heads, 4 KV heads). QK normalization with learned per-head q_gain.
- **leaky_relu²**: `leaky_relu(x, neg_slope=0.01)²` activation in MLP (not GELU).
- **Zero-init**: attn.proj and mlp.proj weights initialized to zero (blocks start as identity).
- **Logit softcap**: `30 × tanh(logits/30)` clamps output logits.
- **Tied embeddings**: tok_emb shared with output projection (default).

### Training Pipeline

- **3 optimizers**: Adam (embeddings), Muon (2D matrix params via Newton-Schulz orthogonalization), Adam (scalar/control params)
- **LR schedule**: Flat at 1.0, then linear warmdown over last 1200 steps (or time-proportional with wallclock cap)
- **Gradient accumulation**: `8 / world_size` micro-batches per step, total 524K tokens/step
- **Warmup**: 20 steps to prime torch.compile, then restore initial weights
- **Serialization**: raw .pt → int8 (or int4) quantize → zlib compress → .ptz → roundtrip eval (scored BPB)

### QAT (Quantization-Aware Training)

When `QAT_BITS > 0`, enables fake quantization (STE) in CastedLinear.forward() after `QAT_START_FRAC` of training. `EXPORT_BITS` controls final quantization precision. int4 uses range [-8,7] stored as int8, which zlib compresses ~2× better.

### Eval-Time Optimization

- `EVAL_SEQ_LEN`: Evaluate at longer context than training (auto-scales RoPE base)
- `TTT_ENABLED`: Test-time training — adapts embeddings + control scalars on val data chunk-by-chunk before scoring
- TTT uses uncompiled base_model (no torch.compile) to avoid shape issues

### Key Environment Variables

**Data**: DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE (must match tokenizer)
**Model**: NUM_LAYERS (default 9), MODEL_DIM (512), NUM_HEADS (8), NUM_KV_HEADS (4), MLP_MULT (2), TIE_EMBEDDINGS (1)
**Depth recurrence**: NUM_UNIQUE_LAYERS (defaults to NUM_LAYERS), NUM_RECURRENCES (1)
**Training**: ITERATIONS (20000), MAX_WALLCLOCK_SECONDS (600), TRAIN_BATCH_TOKENS (524288), TRAIN_SEQ_LEN (1024)
**QAT**: QAT_BITS (0=disabled), QAT_START_FRAC (0.3), EXPORT_BITS (8)
**EMA**: EMA_DECAY (0.0=disabled, recommend 0.9995)
**Eval**: EVAL_SEQ_LEN (=TRAIN_SEQ_LEN), EVAL_STRIDE (0=non-overlapping), TTT_ENABLED (0), TTT_LR (1e-5), EVAL_ROPE_SCALE (0=auto)
**Optimizer**: EMBED_LR, MATRIX_LR (0.04), SCALAR_LR (0.04), MUON_MOMENTUM (0.95)

All defaults are set in `Hyperparameters` class at top of each script.

## Quantization (int8/int4 + zlib)

- 2D float tensors: per-row int8 quantization (clip at 99.99984th percentile, scale per row)
- 1D float tensors: per-tensor int8 (single scale)
- Small tensors (<65K params): keep as float16
- Control tensors (scales, gains, mixes): keep as float32
- Packed → torch.save → zlib.compress(level=9) → `.ptz` file

## Submission Process

PR to `records/track_10min_16mb/<date_name>/` (or `track_non_record_16mb/` for unlimited compute).
Required files: `README.md`, `submission.json`, `train.log`, `train_gpt.py`.
The `train_gpt.py` must compile and run independently within the records folder.

`submission.json` example fields: author, github_id, name, blurb, date, val_loss, val_bpb, bytes_total, bytes_code.

## Participant Info

- Name: Tarkeshwar Narayan Sharma
- GitHub: tns15june
- Fork: https://github.com/tns15june/parameter-golf
- Branch: `submission-v1`
- Submission folder: `records/track_10min_16mb/2026-04_tns15june_v1/`

## Implementation Status

All features (depth recurrence, QAT, eval-time optimization, EMA, sliding-window eval, LeakyReLU²) are implemented in `train_gpt.py` with backward-compatible defaults. Iterated on RunPod 1×H100; final 8×H100 submission run pending — see `dev/run_final.sh` for the production config and `runpod_validate.sh` for the smoke-test pipeline.

### Potential Issues

- torch.compile with fullgraph=True may recompile when QAT enables mid-training (_qat_bits guard change)
- eval_val with eval_seq_len > train_seq_len: trailing val tokens truncated to seq_len multiple
- Depth recurrence in compiled forward: static loop count, round-robin is deterministic — should work

### Optimization Strategy (v4 — after Phase A findings 2026-04-19)

1. **10 layers, 2x MLP**: Depth over width. ~18M params. ~12 MB artifact (fits 16 MB with headroom).
2. **LeakyReLU²**: Better gradient flow than ReLU² (prevents dead neurons)
3. **Int6 QAT at 15%**: Start fake quantization early to minimize export gap
4. **EMA (0.9995)**: Smoother weights → better generalization + compression
5. **Sliding window eval, seq_len=1024 (MATCHES train)**: stride=256 for warm-context. NO RoPE scaling.
6. **lzma compression**: Better ratio than zlib for int6 values

Target config: `NUM_UNIQUE_LAYERS=10 MLP_MULT=2 QAT_BITS=6 QAT_START_FRAC=0.15 EXPORT_BITS=6 EMA_DECAY=0.9995 EVAL_SEQ_LEN=1024 EVAL_STRIDE=256`

### Phase A findings (2026-04-19, 8×H100 runs)

**Run 1 (v2, failed):** Stale GitHub code (9L + n-gram eval). n-gram ran only on rank 0, other ranks blocked on subsequent collective → 10-min NCCL watchdog killed it. Post-export eval never produced a number. ~$12 spent.

**Run 2 (v3, diagnostic):** 10L × MLP=3 × dim=512 × EVAL_SEQ_LEN=2048. Revealed two bugs:
- **Artifact over 16 MB**: 16.27 MB vs 16.00 MB cap (disqualifying)
- **Catastrophic export gap**: 0.57 BPB. pre=1.1919 (at train_seq_len=1024) → post=1.7615 (at eval_seq_len=2048 with RoPE base 10000→20000)
- Root cause: model trained at 1024 cannot extrapolate to 2048 via linear RoPE scaling without position-interpolation training (YaRN/NTK-aware/etc.)

**v4 fix**: MLP_MULT=3→2 (fit 16 MB); EVAL_SEQ_LEN=2048→1024 (eliminate extrapolation). Expected post_bpb ~1.22, export_gap <0.05.

### Next phases (plan, 2026-04-19)

**Phase A retry (v4)** — 8×H100, ~$8, ~15 min. Command:
```
cd /workspace/parameter-golf && git pull && rm -f final_model.int*.ptz && bash dev/runpod_go.sh final 2>&1 | tee /workspace/phase_a_v4.log
```

**Phase B (GPTQ ablation)** — 1×H100, ~$3, ~30 min. Command:
```
bash dev/run_gptq_ablate.sh
```
Compares: amax int6 / GPTQ int6 / GPTQ int4 / GPTQ int4+dim768. Decision:
- If GPTQ int6 gap < amax int6 gap by >0.005: GPTQ works for int6
- If GPTQ int4 gap < 0.015: int4 viable — use freed bytes for dim=768

**Phase C (final submission)** — 8×H100, ~$10. Command based on Phase B winner:
```
bash dev/run_final_gptq.sh [int6|wide]
```

**Phase D stretch (if time/budget allow)** — XSA (cross-sequence attention, what #1 uses). Requires code changes to attention mask + sequence packing.

**Budget so far**: ~$20/$100 spent. Remaining: ~$80 covers v4 ($8) + Phase B ($3) + Phase C ($10) + ~$60 for iteration/stretch goals.

### Still TODO (potential further gains)

- **XSA** (cross-sequence attention): #1 uses this — attention across sequence boundaries
- **Self-Generated GPTQ**: Better per-layer quantization with calibration data (dormant scaffold already in train_gpt.py, enable via `QUANT_METHOD=gptq`)
- **TTT with LoRA**: More expressive test-time training than current embed+scalar TTT
- **Position interpolation training**: Would unlock EVAL_SEQ_LEN>TRAIN_SEQ_LEN path (currently breaks)
- **Tokenizer optimization**: BPE 8192 vocab explored by some submissions
