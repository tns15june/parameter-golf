# FINAL PLAYBOOK — One-Shot Submission

**Budget**: $38. **Deadline**: 2026-04-30. **Strategy**: single 8×H100 pod session, two-phase auto-orchestrated run, auto-commit-and-push between phases.

---

## TL;DR

**One pod. One command. ~$12–14 spent. ~$24 reserve.**

```bash
# On the 8×H100 SXM pod, from /workspace/parameter-golf:
nohup bash dev/ONE_SHOT.sh > logs/ONE_SHOT.log 2>&1 & disown
tail -f logs/ONE_SHOT.log
```

The orchestrator:
1. Downloads SP1024 (if missing) ~$1
2. **Phase 1** — SP1024 v4 trains, quantizes, exports. ~$5. Auto-commits + pushes.
3. Downloads SP8192 from `kevclark/parameter-golf` (pre-tokenized mirror) ~$1
4. **Phase 2** — SP8192 frontier (bigbag-aligned config) trains, quantizes, exports. ~$5.
5. If Phase 2 beats Phase 1 BPB and fits 16 MB → promote, commit, push. Else roll back.

**No 1×H100 prep pod needed.** The old $7 SP8192 retokenize step is bypassed by pulling Kevin Clark's pre-tokenized shards from HF (same path the 1.0810 record uses).

---

## What changed on 2026-04-21 (post code review)

The earlier plan had three expensive / suboptimal elements that I fixed after reading the `train_gpt.py` internals and the bigbag 1.0810 record:

1. **SP8192 data**: was $7 local retokenize on a separate 1×H100 pod. Now pulled pre-tokenized from `kevclark/parameter-golf` on HuggingFace during Phase 2 setup (~$1–2). Same mirror bigbag's reproduction uses.
2. **Frontier recurrence**: was `layers 4..7 × 3 = 19 effective layers`. Now `layers 3..5 × 3 = 17 effective layers`, which produces the **exact** schedule bigbag ships (`[0,1,2,3,4,5,3,4,5,3,4,5,6,7,8,9,10]`). Fewer effective layers → faster steps → more training fits the 600s cap.
3. **Frontier quant/TTT knobs**: `SDCLIP_K` was 2.5 (over-clips int6), now **12.85** (bigbag-tuned). `TTT_LR` was 1e-5 (basically no adaptation), now **5e-3**. `TTT_EPOCHS` 2→3. `TTT_CHUNK_TOKENS` 8192→32768. `MUON_WEIGHT_DECAY` 0.090→0.095. These are the high-leverage knobs responsible for the bulk of the 1.11→1.08 BPB gap in the upstream leaderboard.

The code itself (`train_gpt.py`) was **not edited** — every change is an env-var default in `dev/run_frontier.sh`. If a change blows up, override on the command line (see bottom).

---

## Step 0 — push the working changes from your laptop (free, 1 min)

The three files below are committed locally but haven't been pushed yet:

- `dev/ONE_SHOT.sh` (updated)
- `dev/preflight.sh` (updated)
- `dev/run_frontier.sh` (bigbag-aligned config)
- `dev/runpod_go.sh` (HF-mirror SP8192)
- `FINAL_PLAYBOOK.md` (this file)

```bash
# Local
cd "C:/Users/tarke/Desktop/Parameter Golf/parameter-golf"
git log --oneline -3               # confirm the commits are there
GH_TOKEN=$(gh auth token) git push origin submission-v3-sp8192-full
```

Verify on GitHub that the push landed.

---

## Step 1 — launch the 8×H100 SXM pod and run ONE_SHOT (~$12–14)

```bash
# Launch: https://console.runpod.io/deploy?template=y5cejece4j
# Choose: 8×H100 SXM, 30 GB container disk, 75 GB volume at /workspace, SSH on.

# SSH in:
cd /workspace
[ -d parameter-golf ] || git clone --branch submission-v3-sp8192-full \
    https://github.com/tns15june/parameter-golf.git
cd parameter-golf
git fetch origin && git reset --hard origin/submission-v3-sp8192-full

# (optional) HF token to avoid rate limits during data download:
# echo "<your_hf_token>" > huggingface_token.txt

# Auth gh so ONE_SHOT can auto-push results:
gh auth login   # GitHub.com → HTTPS → paste token or browser flow

# Pre-flight (free)
bash dev/preflight.sh
#   Must print "GREEN". If RED, stop, fix, then rerun.

# THE ONE SHOT — survives SSH disconnect
nohup bash dev/ONE_SHOT.sh > logs/ONE_SHOT.log 2>&1 &
disown
tail -f logs/ONE_SHOT.log
```

Expected wallclock: **~25–30 minutes** (Phase 1 ~12 min incl. SP1024 download, Phase 2 ~15 min incl. SP8192 download).

When you see the final banner `ONE_SHOT — complete`:

```bash
cat records/track_10min_16mb/2026-04_tns15june_v1/submission.json
git log --oneline -5
git status                  # should be clean; commits auto-pushed

# If auto-push failed (log will warn), do it manually:
GH_TOKEN=$(gh auth token) git push origin HEAD
```

**Terminate the pod the moment you're done.** Every minute is $0.42.

---

## Step 2 — open the PR to upstream (free, 5 min)

```bash
# On your laptop
cd "C:/Users/tarke/Desktop/Parameter Golf/parameter-golf"
git fetch origin && git pull origin submission-v3-sp8192-full

gh pr create \
    --repo openai/parameter-golf \
    --base main \
    --head tns15june:submission-v3-sp8192-full \
    --title "Submission: $(python3 -c 'import json; print(json.load(open(\"records/track_10min_16mb/2026-04_tns15june_v1/submission.json\"))[\"name\"])')" \
    --body "See records/track_10min_16mb/2026-04_tns15june_v1/"
```

---

## What the orchestrator does (transparent)

`dev/ONE_SHOT.sh` in default `auto` mode:

**Pre-flight:**
- Asserts 8 GPUs, loads HF token, auto-downloads SP1024 if missing.

**Phase 1 — `dev/run_final.sh` (SP1024 v4 fallback):**
- 10L/512d, MLP 2×, `LeakyReLU²`, int6 QAT, EMA 0.9995, LZMA, sliding-window eval (stride=256, seq_len=1024).
- Log → `logs/submission_v4.txt`. Parses `final_int8_zlib_roundtrip_exact val_bpb:X` and `Total submission size: N bytes`.
- If `bytes ≤ 16,000,000` and `bpb` is finite: calls `dev/fill_submission.py`, `dev/sync_submission.sh`, git-commits, `git push origin HEAD` (via `gh auth token`).

**Data gate:**
- Checks SP8192 shards. If missing, pulls from `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`. If download fails, silently skips Phase 2 (Phase 1 stays as submission).

**Phase 2 — `dev/run_frontier.sh` (SP8192 bigbag-aligned):**
- 11L/512d, MLP 4×, targeted recurrence layers 3–5 × 3 (17 effective), partial RoPE 25%, layerwise RMSNorm scale, QK gain 5.25, parallel residuals (same-block + softmax-mix from later layers), MuonEq-R + WD 0.095, EMA 0.9965.
- GPTQ + SDClip k=12.85 on non-embed matrices at int6, int8 amax on tied embedding. Byte-shuffle + Brotli-11.
- Legal score-first sliding TTT: chunk=32K, epochs=3, lr=5e-3.
- Log → `logs/submission_frontier.txt`.

**Promotion gate:**
- Snapshots Phase 1 artifacts as `*.phase1.bak` before Phase 2 runs.
- If Phase 2 produces `val_bpb < phase1_bpb` AND `bytes ≤ 16,000,000`: overwrites submission.json, train.log, train_gpt.py; commits + pushes.
- Else: restores Phase 1 artifacts from the .bak files. Phase 1 stays as your submission.

**Worst case**: Phase 2 crashes halfway. Phase 1 is already on GitHub from the commit earlier in the run. You still have a legitimate submission.

---

## Failure modes and triage

Read `logs/ONE_SHOT.log` first, then the relevant phase log.

| Symptom | Cause | Action |
|---|---|---|
| Preflight prints `FAIL` on GPU count | Wrong pod | Provision 8×H100 SXM. |
| `SP1024 download failed` | HF rate limit / no token | Put token in `huggingface_token.txt`, restart. |
| Phase 1 `bytes=NNN` where NNN > 16000000 | Compression regression | Inspect log; may need `EXPORT_BITS=5` override. |
| `SP8192 HF download failed. Phase 2 will SKIP.` | HF rate limit / network | Phase 1 still submits. Optionally retry Phase 2 alone: `bash dev/ONE_SHOT.sh frontier`. |
| Phase 2 OOM / NCCL crash | Unknown, never run end-to-end | Phase 1 auto-restored. Inspect `logs/submission_frontier.txt`. Budget for ONE retry: override knobs (see below). |
| Phase 2 `val_bpb` worse than Phase 1 | Tuning miss | Auto rollback. Don't retry blindly. |
| `git push failed` | Pod lost gh auth | `GH_TOKEN=$(gh auth token) git push origin HEAD` manually. |

### Emergency Phase 2 overrides (if first run crashes or regresses)

These reverse individual tuning changes one at a time. Do NOT stack them blindly.

```bash
# Too-aggressive TTT diverged:
TTT_LR=1e-5 TTT_EPOCHS=2 bash dev/run_frontier.sh

# Artifact over 16 MB:
EXPORT_BITS=5 bash dev/run_frontier.sh

# Recurrence caused slow steps / OOM:
RECURRENCE_START_LAYER=4 RECURRENCE_END_LAYER=5 NUM_RECURRENCES=2 bash dev/run_frontier.sh

# SDClip k=12.85 hurt quality (unlikely but possible):
SDCLIP_K=2.5 bash dev/run_frontier.sh

# After a manual retry, if it lands cleanly:
python3 dev/fill_submission.py logs/submission_frontier.txt
bash dev/sync_submission.sh
GH_TOKEN=$(gh auth token) git push origin HEAD
```

---

## Hard rules

1. **Do not edit `train_gpt.py`**. 1500-line ceiling, at the limit. Every change is env-gated.
2. **Do not re-tokenize SP8192 locally**. It's $7 wasted. The HF mirror path is now the default.
3. **Do not retry Phase 2 in a loop**. You have budget for ONE retry max. Each attempt ~$5.
4. **Do not raise `EVAL_SEQ_LEN` above `TRAIN_SEQ_LEN`**. `train_gpt.py` raises RuntimeError; a previous attempt cost 0.57 BPB (see `CLAUDE.md` → Key Learnings).
5. **Do not push to `main` of upstream directly**. Open a PR.

---

## Expected outcomes (honest)

| Scenario | Probability | val_bpb | Leaderboard position |
|---|---:|---:|---|
| Phase 1 lands, Phase 2 lands cleanly at tuned config | ~50% | ~1.09–1.12 | Mid-pack (top 1.0810 unreachable without code edits) |
| Phase 1 lands, Phase 2 crashes → Phase 1 stays | ~40% | ~1.22 | Just above baseline (1.2244), last row |
| Phase 1 crashes too | <10% | — | No submission |

**The top slot (1.0810) is not reachable with this codebase** — bigbag uses progressive recurrence activation, flash_attn_3, and cosine TTT LR decay that our `train_gpt.py` doesn't implement, and we can't add them without risk. Aim: beat baseline cleanly, land mid-pack if lucky.

---

## Budget accounting (revised 2026-04-21)

| Item | Cost | Notes |
|---|---:|---|
| Push local changes to GitHub | $0 | |
| 8×H100 pod startup + preflight | ~$1 | 1–2 min idle |
| SP1024 download (if needed) | ~$1 | ~10 min |
| Phase 1 training + quant + eval | ~$5 | 10 min |
| SP8192 download from HF mirror | ~$1–2 | ~10 min |
| Phase 2 training + quant + eval | ~$5 | 10 min |
| Pod teardown | $0 | |
| **Expected total** | **$12–14** | |
| **Reserve for one retry** | **$24** | Cover one Phase 2 re-run at ~$5 |

---

## Files you care about

| Path | Role |
|---|---|
| `train_gpt.py` | 1500-line core. Do NOT edit. |
| `dev/ONE_SHOT.sh` | **The command you run.** Orchestrator. |
| `dev/preflight.sh` | Pre-run environment check. Run it first. |
| `dev/run_final.sh` | Phase 1 config (SP1024 v4). |
| `dev/run_frontier.sh` | Phase 2 config (SP8192, bigbag-aligned). |
| `dev/runpod_go.sh` | Manual mode launcher (used internally by preflight). |
| `dev/fill_submission.py` | Parses log → fills `submission.json`. |
| `dev/sync_submission.sh` | Copies root `train_gpt.py` → submission folder, verifies sha256. |
| `records/track_10min_16mb/2026-04_tns15june_v1/` | Submission folder. |
| `CLAUDE.md` | Session memory. Read § Key Learnings before anything clever. |
| `FINAL_PLAYBOOK.md` | This file. |
