#!/bin/bash
# =============================================================================
# Parameter Golf — Full-Dataset Validation Pipeline
# =============================================================================
# Validates export pipeline integrity before spending on 8xH100.
# Each experiment uses the FULL dataset and real wallclock budget.
#
# Strategy: repair the scoring pipeline, then optimize the real objective.
#   final_bpb = model_quality + export_gap + eval_gap
#
# Experiment order:
#   1. no_qat_int8      — prove architecture is stable (int8 export)
#   2. no_qat_mixed     — validate mixed-precision export (int4 blocks + int8 embed)
#   3. qat_mixed        — validate QAT reduces int4 export gap
#   4. eval_rope_only   — test RoPE scaling alone
#   5. eval_ttt_only    — test TTT alone
#
# Requirements: RunPod pod with 1×H100+ and 50GB+ disk
# Cost estimate: ~$4-5 on 1×H100 (~65 min total)
# =============================================================================

set -e

RESULTS_FILE="/workspace/parameter-golf/validation_results.txt"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

# Score-aware thresholds
MAX_EXPORT_GAP_INT8=0.02    # int8 gap must be < 0.02 BPB
MAX_EXPORT_GAP_INT4=0.10    # int4 gap must be < 0.10 BPB
MAX_ABSOLUTE_BPB=2.0        # sanity: BPB must be < 2.0

echo "============================================================"
echo "  PARAMETER GOLF — FULL-DATASET VALIDATION"
echo "  $(date)"
echo "  GPUs detected: $NUM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "============================================================"

# ----- STEP 1: Setup -----
echo ""
echo "[1/2] Setting up environment..."

cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone --branch submission-v1 https://github.com/tns15june/parameter-golf.git
else
    cd parameter-golf && git checkout submission-v1 && git pull --ff-only && cd /workspace
fi
cd parameter-golf

pip install -q -r requirements.txt
echo "Dependencies installed."

# ----- STEP 2: Download FULL dataset -----
echo ""
echo "[2/2] Downloading FULL dataset (all shards)..."

TRAIN_SHARD_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$TRAIN_SHARD_COUNT" -ge 80 ]; then
    echo "Full dataset already present ($TRAIN_SHARD_COUNT train shards), skipping."
else
    echo "Downloading... (this takes 5-10 min, ~16GB)"
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi
echo "Dataset ready: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l) train shards"

# ----- Shared submission config (matches run_final.sh) -----
SHARED_CONFIG=(
    NUM_UNIQUE_LAYERS=3
    NUM_RECURRENCES=4
    NUM_LAYERS=12
    MODEL_DIM=768
    NUM_HEADS=12
    NUM_KV_HEADS=6
    MLP_MULT=2
    VOCAB_SIZE=1024
    TRAIN_SEQ_LEN=1024
    TIE_EMBEDDINGS=1
    ROPE_BASE=10000
    LOGIT_SOFTCAP=30.0
    TRAIN_BATCH_TOKENS=524288
    MAX_WALLCLOCK_SECONDS=600
    VAL_LOSS_EVERY=200
    TRAIN_LOG_EVERY=50
)

# ----- Helper function -----
run_experiment() {
    local name="$1"
    local description="$2"
    local gap_threshold="$3"
    local ngpus="$4"
    shift 4
    # Remaining args are KEY=VALUE env overrides

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: $name"
    echo "  $description"
    echo "  GPUs: $ngpus | Gap threshold: $gap_threshold"
    echo "============================================================"

    local start_time=$(date +%s)

    # Build env string from shared config + overrides
    local env_cmd=""
    for kv in "${SHARED_CONFIG[@]}"; do
        env_cmd="$env_cmd $kv"
    done
    for kv in "$@"; do
        env_cmd="$env_cmd $kv"
    done

    # Run with torchrun
    set +e
    env $env_cmd RUN_ID="validate_${name}" \
        torchrun --standalone --nproc_per_node="$ngpus" train_gpt.py \
        2>&1 | tee "/tmp/exp_${name}.log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Parse results from log
    local log_content=$(cat "/tmp/exp_${name}.log")
    local post_bpb=$(echo "$log_content" | grep -oP 'post_export_bpb:\K[\d.]+' | tail -1)
    local pre_bpb=$(echo "$log_content" | grep -oP 'pre_export_bpb:\K[\d.]+' | tail -1)
    local export_gap=$(echo "$log_content" | grep -oP 'export_gap:\K[-\d.]+' | tail -1)
    local params=$(echo "$log_content" | grep -oP 'model_params:\K\d+' | tail -1)
    local compressed=$(echo "$log_content" | grep -oP 'Serialized model int8\+zlib: \K\d+' | tail -1)
    local peak_mem=$(echo "$log_content" | grep -oP 'peak memory allocated: \K\d+' | tail -1)
    local qat_activated=$(echo "$log_content" | grep -c 'QAT enabled at step')
    local rope_scaled=$(echo "$log_content" | grep -c 'RoPE scaled:')
    local ttt_ran=$(echo "$log_content" | grep -c 'Running test-time training eval')

    # Score-aware PASS/FAIL
    local status="FAIL"
    local fail_reasons=""
    if [ "$exit_code" -ne 0 ]; then
        fail_reasons="crashed(exit=$exit_code)"
    elif [ -z "$post_bpb" ]; then
        fail_reasons="no_bpb_in_output"
    else
        # Check absolute BPB sanity (use python3 instead of bc — bc not always installed)
        local bpb_ok=$(python3 -c "print(1 if $post_bpb < $MAX_ABSOLUTE_BPB else 0)" 2>/dev/null || echo "0")
        if [ "$bpb_ok" != "1" ]; then
            fail_reasons="bpb=${post_bpb}>$MAX_ABSOLUTE_BPB"
        fi
        # Check export gap if threshold provided and gap available
        if [ -n "$export_gap" ] && [ "$gap_threshold" != "none" ]; then
            local gap_ok=$(python3 -c "print(1 if abs($export_gap) < $gap_threshold else 0)" 2>/dev/null || echo "0")
            if [ "$gap_ok" != "1" ]; then
                fail_reasons="${fail_reasons:+$fail_reasons,}gap=${export_gap}>${gap_threshold}"
            fi
        fi
        if [ -z "$fail_reasons" ]; then
            status="PASS"
        fi
    fi

    echo ""
    echo "  >> Status: $status ${fail_reasons:+($fail_reasons)}"
    echo "  >> Time: ${elapsed}s"
    [ -n "$pre_bpb" ] && echo "  >> pre_export_bpb=$pre_bpb"
    [ -n "$post_bpb" ] && echo "  >> post_export_bpb=$post_bpb"
    [ -n "$export_gap" ] && echo "  >> export_gap=$export_gap"
    [ -n "$compressed" ] && echo "  >> compressed=${compressed} bytes  params=${params}"
    [ -n "$peak_mem" ] && echo "  >> peak_mem=${peak_mem} MiB"
    [ "$qat_activated" -gt 0 ] && echo "  >> QAT: activated"
    [ "$rope_scaled" -gt 0 ] && echo "  >> RoPE: scaled"
    [ "$ttt_ran" -gt 0 ] && echo "  >> TTT: ran"

    # Append to results file
    echo "$name | $status | pre=$pre_bpb | post=$post_bpb | gap=$export_gap | params=$params | compressed=$compressed | mem=${peak_mem}MiB | ${elapsed}s | qat=$qat_activated rope=$rope_scaled ttt=$ttt_ran${fail_reasons:+ | REASON=$fail_reasons}" >> "$RESULTS_FILE"

    if [ "$exit_code" -ne 0 ]; then
        echo ""
        echo "  >> STDERR (last 30 lines):"
        tail -30 "/tmp/exp_${name}.log" | sed 's/^/    /'
    fi

    return $exit_code
}

# Initialize results file
echo "PARAMETER GOLF VALIDATION RESULTS — $(date)" > "$RESULTS_FILE"
echo "GPUs: $NUM_GPUS" >> "$RESULTS_FILE"
echo "Config: NUM_HEADS=12 NUM_KV_HEADS=6 MODEL_DIM=768 3x4=12eff" >> "$RESULTS_FILE"
echo "---" >> "$RESULTS_FILE"

# =============================================================================
# EXPERIMENT 1: Architecture baseline — no QAT, int8 export
# Purpose: Prove the 12/6-head recurrent architecture is stable.
#          Establish the int8 export gap baseline.
# =============================================================================
echo ""
echo "============================================================"
echo "  [1/5] Architecture baseline (no QAT, int8 export)"
echo "============================================================"

run_experiment "no_qat_int8" \
    "Real submission arch, no QAT, int8 export — establish baseline BPB + gap" \
    "$MAX_EXPORT_GAP_INT8" \
    1 \
    || true

# =============================================================================
# EXPERIMENT 2: Mixed-precision export — int4 blocks + int8 embedding
# Purpose: Validate mixed-precision export without QAT.
#          Embedding (tok_emb) stays int8, block matrices go int4.
#          Should show the raw int4 export penalty on untrained weights.
# =============================================================================
echo ""
echo "============================================================"
echo "  [2/5] Mixed-precision export (int4 blocks + int8 embed, no QAT)"
echo "============================================================"

run_experiment "no_qat_mixed" \
    "No QAT, EXPORT_BITS=4 EMBED_EXPORT_BITS=8 — raw int4 penalty on blocks" \
    "$MAX_EXPORT_GAP_INT4" \
    1 \
    EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
    || true

# =============================================================================
# EXPERIMENT 3: QAT + mixed-precision export
# Purpose: Validate that QAT closes the int4 export gap.
#          This is the core submission config (minus eval tricks).
# =============================================================================
echo ""
echo "============================================================"
echo "  [3/5] QAT + mixed-precision export"
echo "============================================================"

run_experiment "qat_mixed" \
    "QAT_BITS=4 + EXPORT_BITS=4 + EMBED_EXPORT_BITS=8 — QAT should close gap" \
    "$MAX_EXPORT_GAP_INT4" \
    1 \
    QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
    || true

# =============================================================================
# EXPERIMENT 4: RoPE scaling alone (on best training config so far)
# Purpose: Isolate the effect of longer eval context.
#          Uses QAT + mixed export from exp3 as base.
# =============================================================================
echo ""
echo "============================================================"
echo "  [4/5] RoPE 4x context scaling (no TTT)"
echo "============================================================"

run_experiment "eval_rope_only" \
    "QAT mixed + EVAL_SEQ_LEN=4096 — isolate RoPE scaling effect" \
    "none" \
    1 \
    QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
    EVAL_SEQ_LEN=4096 \
    || true

# =============================================================================
# EXPERIMENT 5: TTT alone (on best training config so far)
# Purpose: Isolate the effect of test-time training.
#          Uses QAT + mixed export from exp3 as base.
# =============================================================================
echo ""
echo "============================================================"
echo "  [5/5] Test-time training (no RoPE scaling)"
echo "============================================================"

run_experiment "eval_ttt_only" \
    "QAT mixed + TTT_ENABLED=1 — isolate TTT effect at train_seq_len" \
    "none" \
    1 \
    QAT_BITS=4 QAT_START_FRAC=0.25 EXPORT_BITS=4 EMBED_EXPORT_BITS=8 \
    TTT_ENABLED=1 TTT_LR=1e-5 \
    || true

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo ""
echo "============================================================"
echo "  VALIDATION RESULTS"
echo "============================================================"
echo ""

# Print results table
printf "%-18s %-6s %-10s %-10s %-10s %-12s %-10s %-8s\n" \
    "Experiment" "Status" "Pre BPB" "Post BPB" "Gap" "Compressed" "Peak MiB" "Time"
printf "%-18s %-6s %-10s %-10s %-10s %-12s %-10s %-8s\n" \
    "------------------" "------" "----------" "----------" "----------" "------------" "----------" "--------"

while IFS='|' read -r name status rest; do
    name=$(echo "$name" | xargs)
    status=$(echo "$status" | xargs)

    pre=$(echo "$rest" | grep -oP 'pre=\K[\d.]+' || echo "N/A")
    post=$(echo "$rest" | grep -oP 'post=\K[\d.]+' || echo "N/A")
    gap=$(echo "$rest" | grep -oP 'gap=\K[-\d.]+' || echo "N/A")
    compressed=$(echo "$rest" | grep -oP 'compressed=\K\d+' || echo "N/A")
    mem=$(echo "$rest" | grep -oP 'mem=\K\d+' || echo "N/A")
    elapsed=$(echo "$rest" | grep -oP '\d+s' | head -1 || echo "N/A")

    [ "$name" = "---" ] && continue
    echo "$name" | grep -qE "PARAMETER GOLF|GPUs:|Config:" && continue

    printf "%-18s %-6s %-10s %-10s %-10s %-12s %-10s %-8s\n" \
        "$name" "$status" "$pre" "$post" "$gap" "$compressed" "$mem" "$elapsed"
done < "$RESULTS_FILE"

# Score-based analysis
echo ""
echo "ANALYSIS:"
results_content=$(cat "$RESULTS_FILE")

# Check each experiment
check_experiment() {
    local label="$1"
    local exp_name="$2"
    local line=$(echo "$results_content" | grep "^$exp_name ")
    if [ -z "$line" ]; then
        echo "  [SKIP] $label"
        return
    fi
    local status=$(echo "$line" | cut -d'|' -f2 | xargs)
    if [ "$status" = "PASS" ]; then
        echo "  [PASS] $label"
    else
        local reason=$(echo "$line" | grep -oP 'REASON=\K[^|]+' || echo "unknown")
        echo "  [FAIL] $label ($reason)"
    fi
}

check_experiment "Architecture stable (int8 export)" "no_qat_int8"
check_experiment "Mixed-precision export (int4+int8)" "no_qat_mixed"
check_experiment "QAT closes int4 gap" "qat_mixed"
check_experiment "RoPE scaling" "eval_rope_only"
check_experiment "Test-time training" "eval_ttt_only"

# Size check from best QAT experiment
compressed_bytes=$(echo "$results_content" | grep "qat_mixed" | grep -oP 'compressed=\K\d+' || echo "0")
if [ -f "train_gpt.py" ] && [ "$compressed_bytes" -gt 0 ]; then
    code_bytes=$(wc -c < train_gpt.py)
    total=$((compressed_bytes + code_bytes))
    limit=16000000
    echo ""
    if [ "$total" -lt "$limit" ]; then
        headroom=$((limit - total))
        echo "  [PASS] Size: ${total} bytes (code=${code_bytes} + model=${compressed_bytes}) — ${headroom} bytes headroom"
    else
        echo "  [FAIL] Size: ${total} bytes EXCEEDS 16MB limit!"
    fi
fi

# Decision guide
echo ""
echo "NEXT STEPS:"
echo "  If no_qat_int8 PASS + gap < 0.01:"
echo "    Architecture is solid. Proceed to mixed-precision export."
echo "  If no_qat_mixed PASS + gap < 0.05:"
echo "    Mixed export works. Add QAT to close gap further."
echo "  If qat_mixed PASS + gap < 0.03:"
echo "    Ready for eval tricks. Check rope/ttt for improvement."
echo "  If qat_mixed FAIL (gap too large):"
echo "    Debug export mismatch. Try EXPORT_BITS=8 with bigger model."

echo ""
echo "Full results: $RESULTS_FILE"
echo "Logs: /workspace/parameter-golf/logs/"
echo "============================================================"
echo "Done! $(date)"
