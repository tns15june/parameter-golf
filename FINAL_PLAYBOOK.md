# FINAL PLAYBOOK — One-Shot Submission

**Budget**: $38. **Deadline**: 2026-04-30. **Strategy**: two-phase auto-orchestrated run.

---

## TL;DR

Two separate RunPod pods, in this order:

1. **1×H100 pod** (~$8, optional but strongly recommended for Phase 2): prep SP8192 data.
2. **8×H100 SXM pod** (~$10–15): launch `dev/ONE_SHOT.sh`, which runs Phase 1 (SP1024 v4 guaranteed-baseline-beater), commits + pushes it, then runs Phase 2 (SP8192 frontier) and promotes it only if it beats Phase 1.

**Total cost: ~$18–23 of your $38.** Reserve remainder for one retry only if something goes wrong.

If you must save money, **skip step 1** and ONE_SHOT will auto-run Phase 1 only. You still get a legitimate submission. You just don't take the swing at a top result.

---

## Step 0 — one-time local prep (free)

Push the current branch to GitHub so the pods can clone it:

```bash
# On your local machine
cd "C:/Users/tarke/Desktop/Parameter Golf/parameter-golf"
git status
# If working tree is dirty with untracked/new files from ONE_SHOT.sh etc:
git add dev/ONE_SHOT.sh dev/preflight.sh FINAL_PLAYBOOK.md
git commit -m "Final submission orchestrator + preflight + playbook"
GH_TOKEN=$(gh auth token) git push origin submission-v3-sp8192-full
```

Verify on GitHub that the push landed: <https://github.com/tns15june/parameter-golf/tree/submission-v3-sp8192-full>

---

## Step 1 — SP8192 data prep (1×H100, ~2–3 hr, ~$7)

**SKIP THIS STEP** if any of the following is true:
- You only want a safe sub-baseline submission.
- You already have `/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/` populated with ≥80 train shards from a previous pod session on the same volume.

Otherwise:

1. Launch a **1×H100 PCIe** pod (cheaper than SXM) using the Parameter Golf template:
   <https://console.runpod.io/deploy?template=y5cejece4j>
2. Set Container disk = 30 GB, Volume = 75 GB mounted at `/workspace`. Enable SSH.
3. SSH in and run:

```bash
cd /workspace
rm -rf parameter-golf  # only if a stale copy exists
git clone --branch submission-v3-sp8192-full https://github.com/tns15june/parameter-golf.git
cd parameter-golf
# (Optional) Copy your HF token into the pod if you have one:
#   echo "<your_hf_token>" > huggingface_token.txt

# Validate environment FIRST
bash dev/preflight.sh

# Run prep (2–3 hours, runs in background so SSH disconnects are OK)
nohup bash dev/runpod_go.sh prep-sp8192 > logs/prep_sp8192.log 2>&1 &
disown
tail -f logs/prep_sp8192.log
```

4. When `tail -f` shows it finished (`SP8192 data prep complete`), verify:

```bash
ls data/datasets/fineweb10B_sp8192/fineweb_train_*.bin | wc -l    # expect ≥80
ls data/tokenizers/fineweb_8192_bpe.model                         # expect present
```

5. **IMPORTANT — keep the same pod volume.** Terminate the 1×H100 pod but **save the volume**. When you spin up the 8×H100 pod in Step 2, attach the *same* volume so the SP8192 data is already there.

**If the volume can't be reattached** (e.g., different region), Phase 2 will simply be skipped on the 8×H100 pod — you still get Phase 1. Do NOT re-prep SP8192 on 8×H100 time.

---

## Step 2 — the one-shot run (8×H100 SXM, ~20 min, ~$10–15)

1. Launch an **8×H100 SXM** pod with the Parameter Golf template. Attach the volume from Step 1 if you prepped SP8192.
2. SSH in:

```bash
cd /workspace
# If the repo isn't already on the volume from Step 1:
[ -d parameter-golf ] || git clone --branch submission-v3-sp8192-full https://github.com/tns15june/parameter-golf.git
cd parameter-golf
git fetch origin && git reset --hard origin/submission-v3-sp8192-full

# Copy HF token if you use one
#   echo "<your_hf_token>" > huggingface_token.txt

# Authorize git push-back from the pod (needed for auto-commit of results)
gh auth login   # follow prompts; GitHub → SSH → paste token, OR use browser flow

# Pre-flight
bash dev/preflight.sh
# If this prints "RED", stop and fix before continuing.

# THE ONE SHOT — runs ~20 min, survives SSH disconnect via nohup+disown
nohup bash dev/ONE_SHOT.sh > logs/ONE_SHOT.log 2>&1 &
disown
tail -f logs/ONE_SHOT.log
```

3. Wait for the final banner `ONE_SHOT — complete`. Then:

```bash
# Verify submission.json is populated with real numbers
cat records/track_10min_16mb/2026-04_tns15june_v1/submission.json

# Verify it was pushed to GitHub
git log --oneline -5
git status  # should be clean

# If the auto-push failed (log warned about it), push manually:
GH_TOKEN=$(gh auth token) git push origin HEAD
```

4. **Terminate the pod.** Every minute of 8×H100 uptime is $0.42 burned.

---

## Step 3 — open the PR (free, ~5 min)

Go to <https://github.com/openai/parameter-golf> → "Compare & pull request" on your branch, OR run locally:

```bash
cd "C:/Users/tarke/Desktop/Parameter Golf/parameter-golf"
git fetch origin
git checkout submission-v3-sp8192-full
git pull

# Create the PR
gh pr create \
    --repo openai/parameter-golf \
    --base main \
    --head tns15june:submission-v3-sp8192-full \
    --title "Submission: $(python3 -c 'import json; print(json.load(open("records/track_10min_16mb/2026-04_tns15june_v1/submission.json"))["name"])')" \
    --body "See records/track_10min_16mb/2026-04_tns15june_v1/"
```

---

## What the orchestrator does (for transparency)

`dev/ONE_SHOT.sh` in `auto` mode:

1. **Pre-flight**: verifies 8 GPUs, loads HF token, asserts SP1024 data (auto-downloads if missing), probes SP8192 data (silent skip if missing).
2. **Phase 1** — SP1024 v4 (`dev/run_final.sh`):
   - 10L/512d, MLP 2×, int6 QAT, EMA 0.9995, sliding-window eval, LZMA.
   - Parses `val_bpb` + `bytes_total` from the log.
   - If `bytes_total ≤ 16,000,000` and `val_bpb` is finite: fills `submission.json`, syncs `train_gpt.py`, git-commits and pushes.
3. **Phase 2** — SP8192 frontier (`dev/run_frontier.sh`), only if SP8192 data present:
   - 11L/512d MLP 4× + targeted recurrence (layers 4–7 × 3) + parallel residuals + partial RoPE 25% + layerwise norm scale + QK gain 5.25 + MuonEq-R WD 0.09 + EMA 0.9965 + int6 GPTQ/SDClip + int8 amax embeddings + Brotli-11 byte-shuffle + legal score-first sliding TTT.
   - Snapshots Phase 1 artifacts as `*.phase1.bak` before running.
   - If Phase 2 finishes with `val_bpb < phase1_bpb` AND `bytes ≤ 16 MB`: promotes Phase 2 (overwrites submission.json, pushes).
   - If Phase 2 regresses, crashes, or is over-cap: rolls back to Phase 1 artifacts. Phase 1 stays as your submission.

**Worst case**: Phase 2 crashes halfway. Phase 1 is already on GitHub. You still have a submission.

**Best case**: Phase 2 lands at ~1.09–1.11 BPB, promotes cleanly.

---

## Failure modes and triage

| Symptom (in `logs/ONE_SHOT.log`) | Cause | Action |
|---|---|---|
| `Need 8 GPUs; have N` | Wrong pod | Provision 8×H100 SXM. |
| `SP1024 download failed` | HF rate limit | Add `huggingface_token.txt` to the pod, retry. |
| `Phase 1 produced no usable metrics` | Training crashed | Read `logs/submission_v4.txt` for stack trace. Do not retry blindly. |
| `bytes=NNN` where NNN > 16000000 | Artifact over cap | Phase 1 result rejected. Usually means compression regressed; inspect the log. |
| `SP8192 data NOT ready. Phase 2 will be SKIPPED` | No SP8192 shards on volume | Expected if you skipped Step 1. Phase 1 still runs. |
| `Phase 2 NOT accepted` | Frontier regressed or crashed | Phase 1 was auto-restored. Inspect `logs/submission_frontier.txt`. |
| `git push failed` | Pod lost gh auth | Run `GH_TOKEN=$(gh auth token) git push origin HEAD` manually. |

---

## Hard rules (do NOT violate, even "just to try")

1. **Do not edit `train_gpt.py`**. It is at the 1500-line ceiling. Any unvalidated edit risks the entire run.
2. **Do not re-tokenize SP8192 on an 8×H100 pod.** That's $25/hr × 3 hr = $75, which you don't have.
3. **Do not retry Phase 2 in a loop.** Each 8×H100 attempt is ~$5. You have budget for **one** Phase 2.
4. **Do not change `EVAL_SEQ_LEN` to be larger than `TRAIN_SEQ_LEN`.** The script enforces this, but don't try to work around it. RoPE base scaling without position-interpolation training caused a 0.57 BPB blowup previously (see `CLAUDE.md` → Key Learnings).
5. **Do not push to upstream `main`.** Open a PR, let reviewers merge it.

---

## Budget accounting

| Item | Cost | Notes |
|---|---:|---|
| SP8192 prep (1×H100, 2–3 hr)   | $7  | One-time, optional |
| SP1024 auto-download (if needed) | $1–2 | ~10 min during Phase 1 setup |
| Phase 1 training (10 min)      | $4  | 8×H100 × 10 min |
| Phase 2 training (10 min)      | $4  | 8×H100 × 10 min, only if SP8192 ready |
| Pod setup/teardown overhead    | $2–4 | Per pod session |
| **Expected total**             | **$18–23** | |
| **Remaining reserve**          | **$15–20** | For one emergency retry |

---

## Contact points in the code (for you to reference, NOT edit)

- Entry script: `train_gpt.py` (1500 lines, at ceiling)
- Phase 1 config: `dev/run_final.sh` (SP1024 v4)
- Phase 2 config: `dev/run_frontier.sh` (SP8192 frontier)
- Orchestrator:   `dev/ONE_SHOT.sh` ← the thing you run
- Pre-flight:     `dev/preflight.sh` ← run this first on the pod
- Metrics parser: `dev/fill_submission.py`
- Sync helper:    `dev/sync_submission.sh`
- Submission:     `records/track_10min_16mb/2026-04_tns15june_v1/`
- Session memory: `CLAUDE.md` (read § Key Learnings before doing anything clever)
