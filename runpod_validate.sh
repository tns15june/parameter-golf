#!/bin/bash
# =============================================================================
# Parameter Golf — RunPod GPU Validation Script
# =============================================================================
# Usage: Just paste this entire script into RunPod terminal, or:
#   bash runpod_validate.sh
#
# Requirements: RunPod pod with 1×H100 or 8×H100 (template y5cejece4j)
# Cost estimate: ~$2-3 on 1×H100, ~$5-6 on 8×H100
# =============================================================================

set -e  # Exit on first error

RESULTS_FILE="/workspace/parameter-golf/validation_results.txt"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

echo "============================================================"
echo "  PARAMETER GOLF — RUNPOD VALIDATION"
echo "  $(date)"
echo "  GPUs detected: $NUM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "============================================================"

# ----- STEP 1: Setup -----
echo ""
echo "[1/7] Setting up environment..."

cd /workspace
if [ ! -d "parameter-golf" ]; then
    git clone --branch submission-v1 https://github.com/tns15june/parameter-golf.git
else
    cd parameter-golf && git pull --ff-only && cd /workspace
fi
cd parameter-golf

pip install -q -r requirements.txt 2>&1 | tail -3
echo "Dependencies installed."

# ----- STEP 2: Download data -----
echo ""
echo "[2/7] Downloading dataset..."

if [ -f "data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "Dataset already present, skipping download."
else
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi
echo "Dataset ready."

# ----- Helper function -----
run_experiment() {
    local name="$1"
    local description="$2"
    local ngpus="$3"
    shift 3
    # Remaining args are KEY=VALUE env overrides

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: $name"
    echo "  $description"
    echo "  GPUs: $ngpus"
    echo "============================================================"

    local logfile="logs/kaggle_${name}.txt"

    local start_time=$(date +%s)

    # Build env string
    local env_cmd=""
    for kv in "$@"; do
        env_cmd="$env_cmd $kv"
    done

    # Run with torchrun
    set +e  # Don't exit on experiment failure
    env $env_cmd RUN_ID="validate_${name}" \
        torchrun --standalone --nproc_per_node="$ngpus" train_gpt.py \
        2>&1 | tee "/tmp/exp_${name}.log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Parse results
    local log_content=$(cat "/tmp/exp_${name}.log")
    local val_bpb=$(echo "$log_content" | grep -oP 'final_int8_zlib_roundtrip val_loss:[\d.]+ val_bpb:\K[\d.]+' | tail -1)
    local val_loss=$(echo "$log_content" | grep -oP 'final_int8_zlib_roundtrip val_loss:\K[\d.]+' | tail -1)
    local params=$(echo "$log_content" | grep -oP 'model_params:\K\d+' | tail -1)
    local compressed=$(echo "$log_content" | grep -oP 'Serialized model int8\+zlib: \K\d+' | tail -1)
    local peak_mem=$(echo "$log_content" | grep -oP 'peak memory allocated: \K\d+' | tail -1)
    local qat_activated=$(echo "$log_content" | grep -c 'QAT enabled at step')
    local rope_scaled=$(echo "$log_content" | grep -c 'RoPE scaled:')
    local ttt_ran=$(echo "$log_content" | grep -c 'Running test-time training eval')

    # Status
    local status="FAIL"
    if [ "$exit_code" -eq 0 ]; then
        status="PASS"
    fi

    echo ""
    echo "  >> Result: $status (exit=$exit_code, ${elapsed}s)"
    [ -n "$val_bpb" ] && echo "  >> val_bpb=$val_bpb  val_loss=$val_loss"
    [ -n "$compressed" ] && echo "  >> compressed=${compressed} bytes  params=${params}"
    [ -n "$peak_mem" ] && echo "  >> peak_mem=${peak_mem} MiB"
    [ "$qat_activated" -gt 0 ] && echo "  >> QAT: activated"
    [ "$rope_scaled" -gt 0 ] && echo "  >> RoPE: scaled"
    [ "$ttt_ran" -gt 0 ] && echo "  >> TTT: ran"

    # Append to results file
    echo "$name | $status | bpb=$val_bpb | loss=$val_loss | params=$params | compressed=$compressed | mem=${peak_mem}MiB | ${elapsed}s | qat=$qat_activated rope=$rope_scaled ttt=$ttt_ran" >> "$RESULTS_FILE"

    if [ "$exit_code" -ne 0 ]; then
        echo ""
        echo "  >> STDERR (last 20 lines):"
        tail -20 "/tmp/exp_${name}.log" | sed 's/^/    /'
    fi

    return $exit_code
}

# Initialize results file
echo "PARAMETER GOLF VALIDATION RESULTS — $(date)" > "$RESULTS_FILE"
echo "GPUs: $NUM_GPUS" >> "$RESULTS_FILE"
echo "---" >> "$RESULTS_FILE"

# ----- STEP 3: Smoke test (baseline, ~3 min) -----
echo ""
echo "[3/7] Smoke test: baseline training..."

run_experiment "smoke_test" \
    "Baseline 9 layers, dim=512, 100 iters — validates core training loop" \
    1 \
    ITERATIONS=100 VAL_LOSS_EVERY=100 WARMUP_STEPS=5 TRAIN_LOG_EVERY=20 || true

# ----- STEP 4: Depth recurrence (no QAT, ~10 min) -----
echo ""
echo "[4/7] Depth recurrence: 3 unique × 4 rec = 12 effective, dim=768..."

run_experiment "depth_recurrence" \
    "Weight sharing + wider model — validates recurrence + U-Net skips" \
    1 \
    NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 NUM_LAYERS=12 \
    MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 || true

# ----- STEP 5: QAT int4 (the submission config, ~10 min) -----
echo ""
echo "[5/7] QAT int4: recurrence + fake quantization + int4 export..."

run_experiment "qat_int4" \
    "Full submission config minus eval features — validates QAT + compression" \
    1 \
    NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 NUM_LAYERS=12 \
    MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 \
    QAT_BITS=4 QAT_START_FRAC=0.3 EXPORT_BITS=4 || true

# ----- STEP 6: Eval-time features (RoPE + TTT, ~10 min) -----
echo ""
echo "[6/7] Eval-time features: RoPE scaling + test-time training..."

run_experiment "eval_features" \
    "RoPE 4x context + TTT — validates eval-time optimization" \
    1 \
    NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 NUM_LAYERS=12 \
    MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 \
    QAT_BITS=4 QAT_START_FRAC=0.3 EXPORT_BITS=4 \
    EVAL_SEQ_LEN=4096 TTT_ENABLED=1 TTT_LR=1e-5 || true

# ----- STEP 7: Multi-GPU (if available) -----
if [ "$NUM_GPUS" -ge 2 ]; then
    echo ""
    echo "[7/7] Multi-GPU DDP: $NUM_GPUS GPUs..."

    run_experiment "multi_gpu" \
        "DDP training on ${NUM_GPUS}× GPU — validates distributed code path" \
        "$NUM_GPUS" \
        NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=4 NUM_LAYERS=12 \
        MODEL_DIM=768 NUM_HEADS=8 NUM_KV_HEADS=4 \
        QAT_BITS=4 QAT_START_FRAC=0.3 EXPORT_BITS=4 \
        EVAL_SEQ_LEN=4096 TTT_ENABLED=1 TTT_LR=1e-5 || true
else
    echo ""
    echo "[7/7] SKIP: Multi-GPU (only $NUM_GPUS GPU available)"
    echo "multi_gpu | SKIP | only $NUM_GPUS GPU" >> "$RESULTS_FILE"
fi

# ----- SUMMARY -----
echo ""
echo ""
echo "============================================================"
echo "  FINAL RESULTS SUMMARY"
echo "============================================================"
echo ""

# Print results table
printf "%-20s %-6s %-10s %-10s %-12s %-14s %-10s %-8s\n" \
    "Experiment" "Status" "BPB" "Loss" "Params" "Compressed" "Peak MiB" "Time"
printf "%-20s %-6s %-10s %-10s %-12s %-14s %-10s %-8s\n" \
    "--------------------" "------" "----------" "----------" "------------" "--------------" "----------" "--------"

while IFS='|' read -r name status rest; do
    name=$(echo "$name" | xargs)
    status=$(echo "$status" | xargs)

    bpb=$(echo "$rest" | grep -oP 'bpb=\K[\d.]+' || echo "N/A")
    loss=$(echo "$rest" | grep -oP 'loss=\K[\d.]+' || echo "N/A")
    params=$(echo "$rest" | grep -oP 'params=\K\d+' || echo "N/A")
    compressed=$(echo "$rest" | grep -oP 'compressed=\K\d+' || echo "N/A")
    mem=$(echo "$rest" | grep -oP 'mem=\K\d+' || echo "N/A")
    elapsed=$(echo "$rest" | grep -oP '\d+s' | head -1 || echo "N/A")

    [ "$name" = "---" ] && continue
    echo "$name" | grep -q "PARAMETER GOLF" && continue
    echo "$name" | grep -q "GPUs:" && continue

    printf "%-20s %-6s %-10s %-10s %-12s %-14s %-10s %-8s\n" \
        "$name" "$status" "$bpb" "$loss" "$params" "$compressed" "$mem" "$elapsed"
done < "$RESULTS_FILE"

# Feature checklist
echo ""
echo "FEATURE CHECKLIST:"
results_content=$(cat "$RESULTS_FILE")

check_feature() {
    local label="$1"
    local pattern="$2"
    if echo "$results_content" | grep -q "$pattern"; then
        echo "  [PASS] $label"
    else
        echo "  [FAIL] $label"
    fi
}

check_pass() {
    local label="$1"
    local exp_name="$2"
    if echo "$results_content" | grep "^$exp_name" | grep -q "PASS"; then
        echo "  [PASS] $label"
    else
        echo "  [FAIL] $label"
    fi
}

check_pass "Core training loop + torch.compile" "smoke_test"
check_pass "Depth recurrence (weight sharing)" "depth_recurrence"
check_pass "QAT int4 + zlib compression" "qat_int4"
check_pass "Eval-time features (RoPE + TTT)" "eval_features"

if [ "$NUM_GPUS" -ge 2 ]; then
    check_pass "Multi-GPU DDP" "multi_gpu"
else
    echo "  [SKIP] Multi-GPU DDP (single GPU pod)"
fi

# Size check
compressed_bytes=$(echo "$results_content" | grep "qat_int4" | grep -oP 'compressed=\K\d+' || echo "0")
if [ -f "train_gpt.py" ] && [ "$compressed_bytes" -gt 0 ]; then
    code_bytes=$(wc -c < train_gpt.py)
    total=$((compressed_bytes + code_bytes))
    limit=16000000
    if [ "$total" -lt "$limit" ]; then
        echo "  [PASS] Size budget: ${total} bytes (code=${code_bytes} + model=${compressed_bytes}) < 16MB"
    else
        echo "  [FAIL] Size budget: ${total} bytes EXCEEDS 16MB limit!"
    fi
fi

echo ""
echo "Full results: $RESULTS_FILE"
echo "Logs: /workspace/parameter-golf/logs/"
echo "============================================================"
echo "Done! $(date)"
