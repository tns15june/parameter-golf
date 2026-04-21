# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Overview

**OpenAI Parameter Golf**: Train the best language model (lowest BPB) that fits in a 16MB artifact and trains in ≤10 min on 8×H100 SXM GPUs. Evaluated on FineWeb validation set (first 50K documents), tokenizer-agnostic bits-per-byte.

**Deadline**: 2026-04-30. **Upstream**: https://github.com/openai/parameter-golf. **Our fork**: https://github.com/tns15june/parameter-golf (branch `submission-v1`).

### Key constraints

- Artifact = code bytes + compressed model bytes ≤ 16,000,000 bytes (decimal, not MiB)
- Training ≤10 min on 8×H100. Evaluation ≤10 min (separate budget)
- No network calls or training data access during eval (unless paid for in 16MB)
- New SOTA must beat existing by ≥0.005 nats at p < 0.01
- Tokenizer changes get extra scrutiny; bugs can unjustly improve BPB
- **`train_gpt.py` hard stop: 1500 lines** — currently at the ceiling, so any addition needs an offsetting removal

### Scoring

```
BPB = (cross_entropy_loss / ln(2)) × (tokens_in_sequence / bytes_in_sequence)
```

### Leaderboard status (2026-04-19 upstream audit)

Top upstream run: **1.0810 BPB** (SP8192 + targeted recurrence + parallel residuals + QK-gain 5.25 + legal TTT). Naive baseline: 1.2244. Our submission folder targets the full SP8192 frontier port; the SP1024 v4 stack is kept as a fallback that beats baseline only.

## Commands

### Local smoke (Mac, Apple Silicon)

```bash
cd parameter-golf
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### RunPod (CUDA) — single entry point

Launch pod from template: https://console.runpod.io/deploy?template=y5cejece4j

```bash
cd /workspace
git clone --branch submission-v1 https://github.com/tns15june/parameter-golf.git
cd parameter-golf
```

All orchestrated runs go through `dev/runpod_go.sh <MODE>` — it installs deps, downloads or retokenizes data, reads `huggingface_token.txt` into `$HF_TOKEN` (gitignored), and dispatches:

| Mode | GPUs | Purpose |
|---|---|---|
| `frontier` | 8×H100 | **Current submission** — SP8192 full frontier stack (`dev/run_frontier.sh`) |
| `prep-sp8192` | 1×H100 | One-time SP8192 tokenizer + shard retokenize (~2–3 hr, ~$7). Run before `frontier` |
| `smoke [case]` | 1×H100 | Per-component smokes via `dev/smoke_frontier.sh`; `case=sp8192` exercises SP8192 path |
| `final` | 8×H100 | SP1024 v4 fallback (10L/MLP=2/int6 QAT/LZMA) — beats baseline only |
| `wide` | 8×H100 | SP1024 dim=1024 variant |
| `validate` | 1 GPU | Legacy validation experiments |

Final submission:
```bash
bash dev/runpod_go.sh prep-sp8192                                              # once per volume
bash dev/runpod_go.sh frontier 2>&1 | tee logs/submission_frontier.txt         # ~10 min, ~$4
```

Logs land in `logs/<RUN_ID>.txt`. All model/training params are set via env vars; the two canonical stacks are `dev/run_frontier.sh` (SP8192 frontier) and `dev/run_final.sh` (SP1024 v4).

### Pipeline validation

`runpod_validate.sh` runs a sequence of full-dataset experiments on 1×H100 (~75 min, ~$5–6) to validate the export pipeline (int8 → int6 → +lzma → +ngram → +rope → full stack) before spending on 8×H100. Use after significant changes to serialization/quantization.

### Dataset download

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024                      # full (80 shards, 8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1     # minimal smoke
```

**SP8192 is not published pre-tokenized.** `runpod_go.sh prep-sp8192` retokenizes locally via `data/download_hf_docs_and_tokenize.py --tokenizer-config data/tokenizer_specs_sp8192.json --skip-byte`. Do not pass the combined `tokenizer_specs.json` — it double-processes and hits an unlink-before-reuse bug.

## Architecture (train_gpt.py)

Two scripts: `train_gpt.py` (CUDA/PyTorch, the submission script) and `train_gpt_mlx.py` (MLX, for local Apple Silicon iteration). Both must stay under 1500 lines.

All frontier features are env-gated and default to the SP1024 baseline behavior. See `Hyperparameters` class at the top of `train_gpt.py` for full defaults.

### Base transformer

- **CastedLinear**: weights fp32, cast to bf16 at compute. Supports QAT via `_qat_bits` class variable.
- **GQA**: grouped-query attention (default 8 Q heads, 4 KV). QK normalization with learned per-head `q_gain` (`QK_GAIN_INIT`; 1.5 default, 5.25 frontier — cheap high-leverage knob).
- **leaky_relu²**: `leaky_relu(x, neg_slope=0.01)²` in MLP (not GELU).
- **Zero-init**: `attn.proj` and `mlp.proj` weights start at zero — blocks begin as identity.
- **Logit softcap**: `30 × tanh(logits/30)`.
- **Tied embeddings** (default): `tok_emb` shared with output projection.

### Depth recurrence

Two flavors:
- **Generic** (`TARGETED_RECURRENCE=0`, default): `num_unique_layers` blocks round-robined across `num_effective_layers = unique × recurrences`.
- **Targeted middle-layer recurrence** (`TARGETED_RECURRENCE=1`, frontier): loop only layers `[RECURRENCE_START_LAYER, RECURRENCE_END_LAYER)` `NUM_RECURRENCES` times inside an otherwise-flat backbone. Frontier default: `11L / start=4 / end=7 / recur=3` → 19 effective layers.

Per-effective-layer control scalars (`attn_scales`, `mlp_scales`, `resid_mixes`, `q_gains`, per-layer norm scales) are **not** shared.

### Residual and skip wiring

- **U-Net skip connections**: encoder half stores activations; decoder half adds `skip_weight * stored_activation` from mirror layer.
- **Residual mixing**: each effective layer blends current state with post-embedding `x0` via learned `resid_mix[0]*x + resid_mix[1]*x0`.
- **Parallel residuals** (`PARALLEL_RESIDUALS=1`): attn and MLP both read the same pre-residual state, merged after (~0.01 BPB frontier delta).
- **Parallel later residuals** (`PARALLEL_LATER_RESIDUALS=1`): softmax-weighted mix of all encoder outputs into each decoder layer via learned `late_mix` matrix.
- **Layerwise RMSNorm scale** (`LAYERWISE_NORM_SCALE=1`): per-(effective-)layer learnable scalars on attn and MLP norms.

### RoPE

- `ROPE_FRACTION=1.0` (default): all head dims rotate.
- `ROPE_FRACTION=0.25` (frontier): first 25% of each head's dims rotate; rest pass through unrotated.
- **`EVAL_SEQ_LEN > TRAIN_SEQ_LEN` raises** (`train_gpt.py` ~L1459). See § Key Learnings.

### Training pipeline

- **3 optimizers**:
  - Adam — embeddings (and head, if untied)
  - Muon — 2D matrix params via Newton-Schulz orthogonalization. Optional row-normalization (`MUON_ROW_NORM=1`, MuonEq-R) + weight decay (`MUON_WEIGHT_DECAY=0.09` for frontier).
  - Adam — scalar/control params
- **LR schedule**: flat at 1.0, linear warmdown over last `WARMDOWN_ITERS` (default 1200) steps, time-proportional under wallclock cap.
- **Gradient accumulation**: `8 / world_size` micro-batches per step, total 524K tokens/step.
- **Warmup**: 20 steps to prime `torch.compile`, then restore initial weights.
- **EMA** (`EMA_DECAY>0`): shadow weights swapped in for export (0.9995 SP1024, 0.9965 frontier).
- **Control-surface regularizer** (`CTRL_SURFACE_LAMBDA>0`): amplifies grads on scalar control params.

### Export pipeline

`raw .pt → quantize → compress → .ptz → roundtrip eval`

- **Quantization** (`QUANT_METHOD`):
  - `amax` (default) — per-row int8/int6 on 2D, per-tensor on 1D, percentile-clipped (99.99984th).
  - `gptq` (frontier) — Hessian-aware column rounding calibrated on `GPTQ_CALIB_TOKENS` tokens. `GPTQ_EMBED=1` extends to tied embeddings (off by default — corrupts SP1024 smoke; fixed combined-Hessian not yet validated).
- **SDClip** (`USE_SDCLIP=1`, `SDCLIP_K=2.5`): std-based per-row scale clipping instead of amax/percentile.
- **Precision**: `EXPORT_BITS` for matrices (6 for frontier), `EMBED_EXPORT_BITS` for embeddings (8, amax).
- **Small tensors** (<65K params): kept as float16.
- **Control tensors** (scales, gains, mixes): kept as float32.
- **Compression** (`COMPRESS_METHOD`): `zlib` (legacy), `lzma` (SP1024 v4), `brotli` quality 11 + byte-shuffle (`BYTE_SHUFFLE_STRIDE=2`, frontier).

### QAT

When `QAT_BITS > 0`, `CastedLinear.forward` uses fake quantization (STE) after `QAT_START_FRAC` of training. int4 uses [-8,7] stored as int8 which zlib compresses ~2× better.

**Frontier does not use QAT** — GPTQ + SDClip handles export. QAT is still useful for the SP1024 int6 fallback path (`dev/run_final.sh`).

### Eval-time optimization

- `EVAL_STRIDE < EVAL_SEQ_LEN` (e.g. 256 vs 1024): sliding-window eval for warm context. Only safe lever for context reuse — do **not** raise `EVAL_SEQ_LEN` above `TRAIN_SEQ_LEN` (§ Key Learnings).
- `TTT_ENABLED=1` — test-time training. Uncompiled `base_model` (avoids compile reshape issues).
  - Default: adapts embeddings + control scalars chunk-by-chunk.
  - **Legal score-first sliding TTT** (frontier): score sliding windows for a chunk first, then multi-epoch SGD only on already-scored tokens. `TTT_CHUNK_TOKENS=8192`, `TTT_EPOCHS=2`, `TTT_LR=1e-5`.
- `TTT_ADAPT_ENABLED` — first-order TTT-adaptable training during late training. Uses a manual grad all-reduce (DDP hooks don't fire on the uncompiled outer backward). Off by default — not yet 8×H100-validated.

### Key environment variables

**Data**: `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE` (must match tokenizer)
**Model**: `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `TIE_EMBEDDINGS`, `QK_GAIN_INIT`, `ROPE_BASE`, `ROPE_FRACTION`, `LOGIT_SOFTCAP`
**Recurrence**: `NUM_UNIQUE_LAYERS`, `NUM_RECURRENCES`, `TARGETED_RECURRENCE`, `RECURRENCE_START_LAYER`, `RECURRENCE_END_LAYER`
**Residuals**: `PARALLEL_RESIDUALS`, `PARALLEL_LATER_RESIDUALS`, `LAYERWISE_NORM_SCALE`
**Training**: `ITERATIONS`, `MAX_WALLCLOCK_SECONDS` (600), `TRAIN_BATCH_TOKENS` (524288), `TRAIN_SEQ_LEN` (1024), `WARMDOWN_ITERS`, `WARMUP_STEPS`, `SEED`
**Optimizer**: `EMBED_LR`, `HEAD_LR`, `TIED_EMBED_LR`, `MATRIX_LR`, `SCALAR_LR`, `MUON_MOMENTUM`, `MUON_WEIGHT_DECAY`, `MUON_ROW_NORM`, `BETA1`, `BETA2`, `ADAM_EPS`, `GRAD_CLIP_NORM`
**QAT**: `QAT_BITS` (0=off), `QAT_START_FRAC`
**Export**: `EXPORT_BITS`, `EMBED_EXPORT_BITS`, `QUANT_METHOD` (amax|gptq), `GPTQ_CALIB_TOKENS`, `GPTQ_DAMP_PERCENT`, `GPTQ_EMBED`, `USE_SDCLIP`, `SDCLIP_K`, `COMPRESS_METHOD` (zlib|lzma|brotli), `BYTE_SHUFFLE_STRIDE`
**EMA / reg**: `EMA_DECAY` (0=off), `CTRL_SURFACE_LAMBDA`
**Eval / TTT**: `EVAL_SEQ_LEN`, `EVAL_STRIDE`, `TTT_ENABLED`, `TTT_LR`, `TTT_CHUNK_TOKENS`, `TTT_EPOCHS`, `TTT_MAX_CHUNKS`, `TTT_ADAPT_ENABLED`, `TTT_ADAPT_EVERY`, `TTT_ADAPT_START_FRAC`, `TTT_ADAPT_LR`, `TTT_ADAPT_LAMBDA`

## Submission workflow

**Target folder**: `records/track_10min_16mb/2026-04_tns15june_v1/`

Required files: `README.md`, `submission.json`, `train.log`, `train_gpt.py`. The record's `train_gpt.py` must be byte-identical to root and compile/run inside the record folder.

**After any root `train_gpt.py` change**:
```bash
bash dev/sync_submission.sh   # copies root into submission folder + verifies SHA256
```

**After a training run** (auto-invoked by `dev/run_frontier.sh` / `dev/run_final.sh`):
```bash
python3 dev/fill_submission.py logs/<RUN_ID>.txt
```
Parses `final_int8_zlib_roundtrip_exact` + size lines, populates `submission.json`, copies the log as `train.log`. Picks the name/blurb by detecting config fingerprints in the log (SP8192 frontier / SP1024 frontier / SP1024 v4).

**Git push**: use `GH_TOKEN=$(gh auth token) git push` — plain `git push` fails due to a credential-helper issue on this setup.

## Key Learnings (carry across sessions)

### RoPE base scaling without position-interp training is catastrophic

Training at `TRAIN_SEQ_LEN=N` and evaluating at `EVAL_SEQ_LEN=2N` with linear RoPE base scaling (`base *= 2`) blows up perplexity. **Observed 2026-04-19**: 0.57 BPB export gap (pre=1.1919 at 1024 → post=1.7615 at 2048).

**Why**: linear base scaling shifts ALL position frequencies including short-range ones. Without training-time position interpolation (YaRN, NTK-aware, or training directly at the longer seq_len), attention heads relying on specific short-range frequencies corrupt.

**Rule**: keep `EVAL_SEQ_LEN == TRAIN_SEQ_LEN`. For warm-context benefit use `EVAL_STRIDE < EVAL_SEQ_LEN` (sliding window at same seq_len). `train_gpt.py` enforces this with a RuntimeError.

### Things that have been tried and didn't work

- **int4 QAT** — destroyed training quality in our setup.
- **Late int8 QAT with amax scaling** — worse export gap than no QAT (percentile-based scaling preferred when using amax).
- **TTT + XSA together** — 0.016 BPB mutual harm per upstream competition data.
- **v3 config** (SP1024, MLP_MULT=3, EVAL_SEQ_LEN=2048): 16.27 MB artifact (over cap) + the 0.57 BPB RoPE blowup above.
- **n-gram eval tricks on SP1024**: gain capped at ~0.005 BPB; frontier has moved to SP8192 + architecture instead.

### RunPod operational notes

- Template `y5cejece4j` is the working PyTorch CUDA image; all Python deps pre-installed.
- Container disk 30 GB, volume 75 GB at `/workspace` is sufficient.
- 1×H100 PCIe ~$2.49/hr (ablation); 8×H100 SXM ~$25/hr (final runs).
- First-time SP1024 download ~$4 on 8×H100 (10–15 min); one-time per volume.
- SP8192 retokenize ~2–3 hrs on 1×H100 (~$7); run on a cheap pod first then switch to 8×H100.
- GitHub push required before pod-side `git pull` — pod clones from remote, not local working tree.

## Phase history (for context)

- **Phase A v3 (failed, 2026-04-19)**: SP1024 10L × MLP=3 × EVAL_SEQ_LEN=2048. Killed by artifact-over-cap + RoPE blowup (see Learnings).
- **Phase A v4 (SP1024 fallback)**: SP1024 10L × MLP=2 × EVAL_SEQ_LEN=1024, int6 QAT + LZMA. Expected post_bpb ~1.22. Kept as `dev/run_final.sh`.
- **Frontier port (current)**: full SP8192 stack per `dev/run_frontier.sh`. See `records/track_10min_16mb/2026-04_tns15june_v1/README.md` for the component list.

Budget tracker (2026-04-19 snapshot): ~$20 of $100 RunPod credits spent.

## Potential further gains (not yet tried)

- **XSA** (cross-sequence attention) — attention across sequence boundaries. Used by some older top-3s but skipped in current frontier (TTT is preferred; combining the two is net negative).
- **GPTQ on tied embeddings** (`GPTQ_EMBED=1`) — corrupted SP1024 smoke; needs fixed combined-Hessian for the dual-use weight.
- **First-order TTT-adaptable training** (`TTT_ADAPT_ENABLED=1`) — scaffold shipped with manual grad all-reduce, not yet 8×H100-validated.
- **Position-interpolation training** — would unlock `EVAL_SEQ_LEN > TRAIN_SEQ_LEN`.

## Participant

Tarkeshwar Narayan Sharma · github `tns15june` · fork https://github.com/tns15june/parameter-golf branch `submission-v1` · submission folder `records/track_10min_16mb/2026-04_tns15june_v1/`.
