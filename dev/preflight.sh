#!/bin/bash
# =============================================================================
# dev/preflight.sh — validate environment BEFORE spending any 8xH100 money.
# Run this on any pod (1xH100 is fine) or even locally where it makes sense.
#
# Exit code 0 = green. Non-zero = something will break the paid run.
# =============================================================================
set -u
REPO="${REPO:-/workspace/parameter-golf}"
FAILS=0
WARNS=0

ok()   { echo "  [OK]    $*"; }
warn() { echo "  [WARN]  $*"; WARNS=$((WARNS+1)); }
bad()  { echo "  [FAIL]  $*"; FAILS=$((FAILS+1)); }

echo "=============================================================="
echo "  Parameter Golf — Pre-flight ($(date -u +%FT%TZ))"
echo "=============================================================="

# ---- 1. Repo structure -----------------------------------------------------
echo ""
echo "[1/8] Repo structure"
cd "$REPO" 2>/dev/null || { bad "Repo not at $REPO"; echo "Abort."; exit 1; }
ok "repo at $REPO"
[ -f train_gpt.py ]                      && ok "train_gpt.py present"      || bad "train_gpt.py missing"
[ -f dev/run_final.sh ]                  && ok "dev/run_final.sh present"  || bad "dev/run_final.sh missing"
[ -f dev/run_frontier.sh ]               && ok "dev/run_frontier.sh present" || bad "dev/run_frontier.sh missing"
[ -f dev/ONE_SHOT.sh ]                   && ok "dev/ONE_SHOT.sh present"   || bad "dev/ONE_SHOT.sh missing"
[ -d records/track_10min_16mb/2026-04_tns15june_v1 ] && ok "submission folder present" || bad "submission folder missing"

# train_gpt.py must be exactly the 1500-line ceiling or below
LC=$(wc -l < train_gpt.py)
if [ "$LC" -le 1500 ]; then ok "train_gpt.py line count: $LC (≤1500)"; else bad "train_gpt.py over 1500 lines: $LC"; fi

# ---- 2. Script sync --------------------------------------------------------
echo ""
echo "[2/8] submission/train_gpt.py sync"
SUB_SCRIPT=records/track_10min_16mb/2026-04_tns15june_v1/train_gpt.py
if [ -f "$SUB_SCRIPT" ] && cmp -s train_gpt.py "$SUB_SCRIPT"; then
    ok "submission train_gpt.py is byte-identical to root"
else
    warn "submission train_gpt.py OUT OF SYNC — will be fixed by dev/sync_submission.sh after run"
fi

# ---- 3. GPU visibility -----------------------------------------------------
echo ""
echo "[3/8] GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    NGPUS=$(nvidia-smi -L | wc -l)
    ok "nvidia-smi sees $NGPUS GPU(s)"
    nvidia-smi -L | sed 's/^/        /'
    if [ "$NGPUS" -ge 8 ]; then
        ok "enough GPUs for 8xH100 run"
    elif [ "$NGPUS" -ge 1 ]; then
        warn "only $NGPUS GPU — OK for prep/smoke, NOT enough for final run"
    fi
else
    warn "no nvidia-smi (are you running this locally? that's fine, just skip this check on the real pod)"
fi

# ---- 4. Python / torch -----------------------------------------------------
echo ""
echo "[4/8] Python deps"
python3 -c "import torch; print('        torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 \
    && ok "torch importable" || bad "torch import failed"
python3 -c "import sentencepiece, numpy, brotli, zlib, lzma" 2>&1 \
    && ok "sentencepiece/numpy/brotli/zlib/lzma OK" \
    || warn "some deps missing — run: pip install -r requirements.txt"

# ---- 5. Data ---------------------------------------------------------------
echo ""
echo "[5/8] Data"
SP1024_DIR=data/datasets/fineweb10B_sp1024
SP8192_DIR=data/datasets/fineweb10B_sp8192
SP8192_MODEL=data/tokenizers/fineweb_8192_bpe.model

SP1024_N=$(ls "$SP1024_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
SP1024_VAL=$(ls "$SP1024_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
if [ "$SP1024_N" -ge 80 ] && [ "$SP1024_VAL" -ge 1 ]; then
    ok "SP1024: $SP1024_N train + $SP1024_VAL val shard (Phase 1 READY)"
else
    warn "SP1024: $SP1024_N train, $SP1024_VAL val — ONE_SHOT will auto-download (~10 min, $1–2 on 8xH100)"
fi

SP8192_N=$(ls "$SP8192_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
SP8192_VAL=$(ls "$SP8192_DIR"/fineweb_val_*.bin 2>/dev/null | wc -l)
if [ "$SP8192_N" -ge 80 ] && [ "$SP8192_VAL" -ge 1 ] && [ -f "$SP8192_MODEL" ]; then
    ok "SP8192: $SP8192_N train + $SP8192_VAL val shard + tokenizer (Phase 2 READY)"
else
    warn "SP8192: $SP8192_N train, $SP8192_VAL val, model=$([ -f "$SP8192_MODEL" ] && echo yes || echo NO) — Phase 2 will SKIP unless prepped on a cheap 1xH100 first (~\$7, 2–3 hr)"
fi

# ---- 6. Disk ---------------------------------------------------------------
echo ""
echo "[6/8] Disk"
df -BG /workspace 2>/dev/null | tail -1 | awk '{print "        /workspace:", $2,"total,",$3,"used,",$4,"free"}'
FREE_GB=$(df -BG /workspace 2>/dev/null | awk 'NR==2{sub("G","",$4); print $4}')
if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -ge 30 ]; then
    ok "≥30 GB free on /workspace"
elif [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -ge 10 ]; then
    warn "only ${FREE_GB}G free — enough for SP1024, tight for SP8192 prep"
else
    bad "insufficient disk: ${FREE_GB:-?}G"
fi

# ---- 7. Git / push ---------------------------------------------------------
echo ""
echo "[7/8] Git"
git rev-parse --abbrev-ref HEAD 2>/dev/null | awk '{print "        branch:", $0}'
git rev-parse HEAD 2>/dev/null | awk '{print "        HEAD:  ", $0}'
if git remote get-url origin >/dev/null 2>&1; then
    ok "git origin: $(git remote get-url origin)"
else
    bad "no git origin configured"
fi
# check for uncommitted changes
if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
    ok "working tree clean"
else
    warn "working tree dirty — ONE_SHOT will commit all changes at phase end"
fi
# check gh auth
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    ok "gh auth OK (ONE_SHOT can auto-push)"
else
    warn "gh not authed — ONE_SHOT will commit locally; you must push manually with:"
    echo "          GH_TOKEN=\$(gh auth token) git push origin HEAD"
fi

# ---- 8. Tokens -------------------------------------------------------------
echo ""
echo "[8/8] Tokens"
if [ -n "${HF_TOKEN:-}" ]; then
    ok "HF_TOKEN set in env"
elif [ -f huggingface_token.txt ]; then
    CHARS=$(tr -d '[:space:]' < huggingface_token.txt | wc -c)
    if [ "$CHARS" -gt 10 ]; then
        ok "huggingface_token.txt present ($CHARS chars) — ONE_SHOT will load it"
    else
        warn "huggingface_token.txt looks empty/short"
    fi
else
    warn "no HF_TOKEN — SP1024 download may hit rate limits"
fi

# ---- Summary ---------------------------------------------------------------
echo ""
echo "=============================================================="
echo "  Pre-flight result: $FAILS fail(s), $WARNS warning(s)"
echo "=============================================================="
if [ "$FAILS" -eq 0 ]; then
    echo "  GREEN. Safe to launch: nohup bash dev/ONE_SHOT.sh > logs/ONE_SHOT.log 2>&1 & disown"
    exit 0
else
    echo "  RED. Fix failures before spending 8xH100 time."
    exit 1
fi
