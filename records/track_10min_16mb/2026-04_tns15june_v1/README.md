## SP8192 Frontier: Full Port

### Approach

SP8192-tokenized FineWeb with the full frontier component set. Nine interlocking
pieces, not a grab-bag of grafts:

1. **SP8192 tokenizer + data**: locally trained SentencePiece BPE 8192 over
   `docs_selected.jsonl`; 196 retokenized shards (195 train + 1 val).
2. **11 layers × 512 dim × MLP 4x**: deeper, wider MLP than the SP1024 scaffold.
3. **Partial RoPE (25%)**: first 25% of each head's dims rotate; 75% pass through.
4. **Layerwise RMSNorm scale**: per-(effective-)layer learnable scalars on both
   attention and MLP norms.
5. **Targeted middle-layer recurrence**: layers 4..7 looped 3× inside an
   11-layer physical backbone (19 effective layers). Per-effective-layer
   control scalars unshared.
6. **Parallel residuals from later layers**: softmax-weighted mix of all encoder
   outputs into each decoder layer via learned `late_mix` matrix.
7. **Optimizer**: Muon with row-normalization (MuonEq-R) + weight decay 0.09;
   Adam for embeddings/heads/scalars. EMA decay 0.9965 for export weights.
8. **Export**: pure GPTQ (Hessian-aware column rounding) on non-embed 2D weights
   with SDClip per-row scale (k=2.5). int8 amax embeddings. Byte-shuffle +
   Brotli quality-11 compression.
9. **Eval**: legal score-first sliding TTT (`TTT_ENABLED=1`) with scope-once-
   per-token accumulation matching `eval_val_sliding`'s protocol.

### Configuration

See `dev/run_frontier.sh` for the complete env-var set. Key defaults:

```
VOCAB_SIZE=8192 DATA_PATH=data/datasets/fineweb10B_sp8192
NUM_UNIQUE_LAYERS=11 NUM_RECURRENCES=3 TARGETED_RECURRENCE=1
RECURRENCE_START_LAYER=4 RECURRENCE_END_LAYER=7
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4
TRAIN_SEQ_LEN=1024 TIE_EMBEDDINGS=1
QK_GAIN_INIT=5.25 ROPE_FRACTION=0.25
PARALLEL_RESIDUALS=1 PARALLEL_LATER_RESIDUALS=1 LAYERWISE_NORM_SCALE=1
MUON_WEIGHT_DECAY=0.09 MUON_ROW_NORM=1
EXPORT_BITS=6 EMBED_EXPORT_BITS=8 QUANT_METHOD=gptq USE_SDCLIP=1 SDCLIP_K=2.5
EMA_DECAY=0.9965 EVAL_STRIDE=256 TTT_ENABLED=1 TTT_CHUNK_TOKENS=8192 TTT_EPOCHS=2
COMPRESS_METHOD=brotli BYTE_SHUFFLE_STRIDE=2
TRAIN_BATCH_TOKENS=524288 MAX_WALLCLOCK_SECONDS=600
```

### Data Prep (one-time on 1×H100, ~2–3 hrs, ~$7)

SP8192 is **not** published pre-tokenized by the upstream manifest. Produce it
locally:

```bash
python3 data/download_hf_docs_and_tokenize.py \
    --output-root data --tokenizer-config data/tokenizer_specs.json \
    --skip-byte --reuse-sp-model 1024=data/tokenizers/fineweb_1024_bpe.model
```

The tokenizer spec is in `data/tokenizer_specs.json` under `sp_bpe_8192`.

### Run

```bash
bash dev/runpod_go.sh frontier 2>&1 | tee logs/submission_frontier.txt
```

The run script auto-invokes `dev/fill_submission.py` to populate
`submission.json` with real metrics + frontier blurb.

### Key Metrics

*(Filled post-run. 2-seed mean: seed 1337 + SEED=42.)*

- Pre-quant eval: `val_loss:___ val_bpb:___`
- Post-quant roundtrip: `val_loss:___ val_bpb:___`
- Export gap: `___`
- Train time: `___ ms`
- Eval time: `___ ms`
- Serialized model (int6 + Brotli + byte-shuffle): `___ bytes`
- Code size: `___ bytes`
- Total submission size: `___ bytes`

### Included Files

- `train_gpt.py` — byte-identical copy of root (enforced by `dev/sync_submission.sh`)
- `train.log` — full log from the 8×H100 run
- `submission.json` — auto-filled metrics + matching SP8192 blurb

### Deferred (known gaps vs current frontier record)

- Global-causal-order legal TTT (current impl is rank-partitioned; scope-once
  fix landed but cross-rank adaptation stream does not propagate)
- GPTQ on tied embeddings (requires fixed combined Hessian for the dual-use
  weight; our post-hook captures only the F.linear path and corrupted the
  embedding lookup in SP1024 smoke — currently `GPTQ_EMBED=0`)
- TTT-adaptable training (`TTT_ADAPT_ENABLED=0` by default — uncompiled backward
  requires the manual gradient all-reduce shipped in this branch but not yet
  empirically validated at 8×H100)
