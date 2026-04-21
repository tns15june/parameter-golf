#!/bin/bash
# =============================================================================
# dev/ONE_SHOT.sh — Final submission orchestrator (run on 8xH100 pod)
# =============================================================================
#
# Two-phase strategy with auto-fallback and auto-commit:
#
#   Phase 1  SP1024 v4 (dev/run_final.sh)        ~10 min, ~$5
#            Guaranteed legitimate submission. Locks in baseline-beating result.
#            On success: fills submission.json, syncs, commits, pushes to GitHub.
#
#   Phase 2  SP8192 frontier (dev/run_frontier.sh)  ~10 min, ~$5–7
#            Only runs if SP8192 data already present on /workspace.
#            On success with LOWER val_bpb than Phase 1: overwrites submission,
#            commits, pushes. On regression or crash: Phase 1 result stays.
#
# Net spend: ~$10–15 on 8xH100. SP8192 prep (~$7 on 1xH100, 2–3 hr) must have
# been done previously on a cheap pod — this script refuses to retokenize on
# 8xH100 time.
#
# ---- USAGE (on 8xH100 pod, survives SSH disconnect) ------------------------
#
#   cd /workspace/parameter-golf
#   git fetch origin && git checkout submission-v3-sp8192-full && git pull
#   nohup bash dev/ONE_SHOT.sh > logs/ONE_SHOT.log 2>&1 &
#   disown
#   tail -f logs/ONE_SHOT.log
#
# Modes (optional 1st arg):
#   auto   (default) — Phase 1, then Phase 2 iff SP8192 data exists
#   safe             — Phase 1 only
#   frontier         — Phase 2 only (Phase 1 skipped; use ONLY if you've
#                      already got Phase 1 landed on another session)
#
# =============================================================================

set -u  # error on unset vars; do NOT use -e (we want to survive phase failures)
MODE="${1:-auto}"

# ---- constants -------------------------------------------------------------
REPO=/workspace/parameter-golf
SP8192_DIR="$REPO/data/datasets/fineweb10B_sp8192"
SP1024_DIR="$REPO/data/datasets/fineweb10B_sp1024"
SUBMISSION_DIR="$REPO/records/track_10min_16mb/2026-04_tns15june_v1"
# NOTE: run_final.sh hardcodes RUN_ID=submission_v4 inline (overrides env).
# run_frontier.sh respects ${RUN_ID:-submission_frontier}. We match those.
PHASE1_LOG="$REPO/logs/submission_v4.txt"
PHASE2_LOG="$REPO/logs/submission_frontier.txt"
STATE_FILE="$REPO/logs/ONE_SHOT.state"

mkdir -p "$REPO/logs"

banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "  $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "============================================================"
}

fail() { echo "FATAL: $*" >&2; exit 2; }

extract_bpb() {
    # $1 = log file. echoes val_bpb (8-decimal) from roundtrip_exact line.
    local log="$1"
    [ -f "$log" ] || { echo ""; return; }
    grep -E "final_int8_zlib_roundtrip_exact" "$log" \
        | tail -1 \
        | sed -E 's/.*val_bpb:([0-9.]+).*/\1/'
}

extract_bytes_total() {
    local log="$1"
    [ -f "$log" ] || { echo ""; return; }
    grep -E "Total submission size" "$log" \
        | tail -1 \
        | sed -E 's/.*: ([0-9]+) bytes.*/\1/'
}

git_commit_and_push() {
    local msg="$1"
    cd "$REPO"
    git add -A records/track_10min_16mb/2026-04_tns15june_v1/ train_gpt.py logs/*.txt 2>/dev/null || true
    # skip if nothing staged
    if git diff --cached --quiet; then
        echo "git: nothing to commit"
        return 0
    fi
    git -c user.email="tns15june@users.noreply.github.com" \
        -c user.name="Tarkeshwar Narayan Sharma" \
        commit -m "$msg" || return 1
    # Push — CLAUDE.md says plain `git push` fails here; use gh token.
    local token=""
    if command -v gh >/dev/null 2>&1; then
        token="$(gh auth token 2>/dev/null || true)"
    fi
    if [ -n "$token" ]; then
        GH_TOKEN="$token" git push origin HEAD || {
            echo "WARN: git push failed — commit is local. Push manually later."
            return 1
        }
    else
        git push origin HEAD || {
            echo "WARN: git push failed and no gh token. Run manually:"
            echo "      GH_TOKEN=\$(gh auth token) git push origin HEAD"
            return 1
        }
    fi
    echo "git: pushed."
    return 0
}

# =============================================================================
# PRE-FLIGHT
# =============================================================================
banner "ONE_SHOT — pre-flight checks"

NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "GPUs detected: $NGPUS"
[ "$NGPUS" -ge 8 ] || fail "Need 8 GPUs; have $NGPUS. Provision an 8xH100 SXM pod."

[ -d "$REPO" ] || fail "Repo not at $REPO. Did you clone to /workspace/parameter-golf?"
cd "$REPO"

# Load HF token if present (some data paths need it)
if [ -z "${HF_TOKEN:-}" ] && [ -f huggingface_token.txt ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < huggingface_token.txt)"
    echo "HF_TOKEN loaded ($(echo -n "$HF_TOKEN" | wc -c) chars)"
fi

# Deps — should be pre-installed on pgut template but cheap to re-assert
pip install -q -r requirements.txt || fail "pip install failed"

# SP1024 data for Phase 1
SP1024_SHARDS=$(ls "$SP1024_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
echo "SP1024 train shards: $SP1024_SHARDS"
if [ "$SP1024_SHARDS" -lt 80 ]; then
    echo "  SP1024 data incomplete — downloading (one-time, ~10 min)..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 \
        || fail "SP1024 download failed"
    SP1024_SHARDS=$(ls "$SP1024_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
fi
[ "$SP1024_SHARDS" -ge 80 ] || fail "SP1024 shards still short ($SP1024_SHARDS)"

# SP8192 data for Phase 2 (presence check only — do NOT generate on 8xH100)
SP8192_SHARDS=$(ls "$SP8192_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
SP8192_MODEL="$REPO/data/tokenizers/fineweb_8192_bpe.model"
echo "SP8192 train shards: $SP8192_SHARDS  (model exists: $([ -f "$SP8192_MODEL" ] && echo yes || echo NO))"

RUN_PHASE2=0
if [ "$MODE" != "safe" ]; then
    if [ "$SP8192_SHARDS" -ge 80 ] && [ -f "$SP8192_MODEL" ]; then
        RUN_PHASE2=1
    else
        echo "  SP8192 data NOT ready. Phase 2 will be SKIPPED."
        echo "  To enable, run this ONCE on a cheap 1xH100 pod first:"
        echo "      bash dev/runpod_go.sh prep-sp8192   # ~2–3 hr, ~\$7"
        RUN_PHASE2=0
    fi
fi

# Disk check — 8B token shards are ~16 GB each variant
DF_GB=$(df -BG /workspace 2>/dev/null | awk 'NR==2{sub("G","",$4); print $4}')
echo "Free disk on /workspace: ${DF_GB:-?} GB"

# Git identity — needed for auto-commit
git config user.email >/dev/null 2>&1 || git config user.email "tns15june@users.noreply.github.com"
git config user.name  >/dev/null 2>&1 || git config user.name  "Tarkeshwar Narayan Sharma"

echo "Pre-flight OK."
echo "Plan:"
case "$MODE" in
    safe)      echo "  MODE=safe      → Phase 1 only" ;;
    frontier)  echo "  MODE=frontier  → Phase 2 only (Phase 1 skipped!)" ;;
    auto)
        if [ "$RUN_PHASE2" = "1" ]; then
            echo "  MODE=auto      → Phase 1, then Phase 2 (SP8192 ready)"
        else
            echo "  MODE=auto      → Phase 1 only (SP8192 not ready — Phase 2 skipped)"
        fi ;;
    *) fail "Unknown mode: $MODE (use auto|safe|frontier)" ;;
esac

PHASE1_BPB=""
PHASE2_BPB=""

# =============================================================================
# PHASE 1 — SP1024 v4 fallback
# =============================================================================
if [ "$MODE" != "frontier" ]; then
    banner "PHASE 1 — SP1024 v4 fallback (guaranteed submission)"

    # run_final.sh writes its own log to logs/submission_v4.txt (=PHASE1_LOG).
    # Also mirror combined stdout+stderr to a stream log for debugging.
    bash dev/run_final.sh 2>&1 | tee "$REPO/logs/ONE_SHOT.phase1.stream.log"
    P1_STATUS=${PIPESTATUS[0]}

    if [ "$P1_STATUS" -ne 0 ]; then
        echo "WARN: Phase 1 run_final.sh exited $P1_STATUS"
    fi

    PHASE1_BPB=$(extract_bpb "$PHASE1_LOG")
    PHASE1_BYTES=$(extract_bytes_total "$PHASE1_LOG")
    echo "Phase 1 val_bpb: ${PHASE1_BPB:-MISSING}"
    echo "Phase 1 bytes:   ${PHASE1_BYTES:-MISSING}"

    if [ -n "$PHASE1_BPB" ] && [ -n "$PHASE1_BYTES" ] && [ "$PHASE1_BYTES" -le 16000000 ]; then
        echo "Phase 1 SUCCESS — filling submission.json + syncing + committing"
        python3 dev/fill_submission.py "$PHASE1_LOG" || echo "WARN: fill_submission failed"
        bash dev/sync_submission.sh || echo "WARN: sync_submission failed"
        echo "phase1_bpb=$PHASE1_BPB" >> "$STATE_FILE"
        echo "phase1_bytes=$PHASE1_BYTES" >> "$STATE_FILE"
        git_commit_and_push "Phase 1 locked: SP1024 v4 val_bpb=$PHASE1_BPB bytes=$PHASE1_BYTES"
    else
        echo "ERROR: Phase 1 produced no usable metrics or oversize artifact."
        echo "  bpb=$PHASE1_BPB  bytes=$PHASE1_BYTES"
        echo "  Submission.json NOT updated. Inspect $PHASE1_LOG."
    fi
fi

# =============================================================================
# PHASE 2 — SP8192 frontier
# =============================================================================
if [ "$RUN_PHASE2" = "1" ] || [ "$MODE" = "frontier" ]; then
    banner "PHASE 2 — SP8192 frontier"

    # Snapshot Phase 1 submission.json so we can roll back on regression.
    SUB_JSON="$SUBMISSION_DIR/submission.json"
    SUB_LOG="$SUBMISSION_DIR/train.log"
    SUB_SCRIPT="$SUBMISSION_DIR/train_gpt.py"
    cp -f "$SUB_JSON"   "$SUB_JSON.phase1.bak"   2>/dev/null || true
    cp -f "$SUB_LOG"    "$SUB_LOG.phase1.bak"    2>/dev/null || true
    cp -f "$SUB_SCRIPT" "$SUB_SCRIPT.phase1.bak" 2>/dev/null || true

    # Let run_frontier.sh use its default RUN_ID=submission_frontier so its
    # internal log path matches PHASE2_LOG.
    unset RUN_ID
    bash dev/run_frontier.sh 2>&1 | tee "$REPO/logs/ONE_SHOT.phase2.stream.log"
    P2_STATUS=${PIPESTATUS[0]}

    if [ "$P2_STATUS" -ne 0 ]; then
        echo "WARN: Phase 2 run_frontier.sh exited $P2_STATUS"
    fi

    PHASE2_BPB=$(extract_bpb "$PHASE2_LOG")
    PHASE2_BYTES=$(extract_bytes_total "$PHASE2_LOG")
    echo "Phase 2 val_bpb: ${PHASE2_BPB:-MISSING}"
    echo "Phase 2 bytes:   ${PHASE2_BYTES:-MISSING}"

    ACCEPT=0
    if [ -n "$PHASE2_BPB" ] && [ -n "$PHASE2_BYTES" ] && [ "$PHASE2_BYTES" -le 16000000 ]; then
        if [ -n "$PHASE1_BPB" ]; then
            # Accept if PHASE2_BPB < PHASE1_BPB (lower is better)
            if awk -v a="$PHASE2_BPB" -v b="$PHASE1_BPB" 'BEGIN{exit !(a<b)}'; then
                ACCEPT=1
            fi
        else
            # No Phase 1 baseline to compare — accept any valid result
            ACCEPT=1
        fi
    fi

    if [ "$ACCEPT" = "1" ]; then
        echo "Phase 2 SUCCESS and BEATS Phase 1 — promoting."
        python3 dev/fill_submission.py "$PHASE2_LOG" || echo "WARN: fill_submission failed"
        bash dev/sync_submission.sh || echo "WARN: sync_submission failed"
        echo "phase2_bpb=$PHASE2_BPB" >> "$STATE_FILE"
        echo "phase2_bytes=$PHASE2_BYTES" >> "$STATE_FILE"
        echo "accepted=phase2" >> "$STATE_FILE"
        git_commit_and_push "Phase 2 promoted: SP8192 frontier val_bpb=$PHASE2_BPB bytes=$PHASE2_BYTES (beat Phase 1 ${PHASE1_BPB:-n/a})"
    else
        echo "Phase 2 NOT accepted (regressed, crashed, or over-cap). Rolling back to Phase 1."
        [ -f "$SUB_JSON.phase1.bak" ]   && mv -f "$SUB_JSON.phase1.bak"   "$SUB_JSON"
        [ -f "$SUB_LOG.phase1.bak" ]    && mv -f "$SUB_LOG.phase1.bak"    "$SUB_LOG"
        [ -f "$SUB_SCRIPT.phase1.bak" ] && mv -f "$SUB_SCRIPT.phase1.bak" "$SUB_SCRIPT"
        echo "accepted=phase1" >> "$STATE_FILE"
    fi

    # Clean up any leftover backups on success
    rm -f "$SUB_JSON.phase1.bak" "$SUB_LOG.phase1.bak" "$SUB_SCRIPT.phase1.bak"
fi

# =============================================================================
# FINAL REPORT
# =============================================================================
banner "ONE_SHOT — complete"
echo "Phase 1 val_bpb: ${PHASE1_BPB:-skipped/failed}"
echo "Phase 2 val_bpb: ${PHASE2_BPB:-skipped/failed}"
echo ""
echo "Current submission.json:"
if [ -f "$SUBMISSION_DIR/submission.json" ]; then
    python3 -c "import json; d=json.load(open('$SUBMISSION_DIR/submission.json')); print(f'  val_bpb:     {d.get(\"val_bpb\")}'); print(f'  bytes_total: {d.get(\"bytes_total\")}'); print(f'  beats baseline by: {1.2244 - float(d.get(\"val_bpb\",0)):.4f} BPB')"
fi
echo ""
echo "Logs:"
echo "  Phase 1: $PHASE1_LOG"
echo "  Phase 2: $PHASE2_LOG"
echo "  State:   $STATE_FILE"
echo ""
echo "NEXT STEPS (manual):"
echo "  1. Verify the submission folder contents look right:"
echo "       ls -la $SUBMISSION_DIR/"
echo "       cat $SUBMISSION_DIR/submission.json"
echo "  2. Confirm git push landed on GitHub (fork branch)."
echo "  3. Open a PR to openai/parameter-golf from your fork branch."
echo "     Target: records/track_10min_16mb/2026-04_tns15june_v1/"
echo ""
