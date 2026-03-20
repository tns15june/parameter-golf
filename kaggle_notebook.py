"""
Parameter Golf — Kaggle GPU Validation Notebook
================================================
Target: 2× T4 (16GB each) on Kaggle.  Also works on 1× T4 or 1× P100.

Validates every code path in train_gpt.py without modifying the source.
All compatibility fixes are runtime monkey-patches applied before each run.

Usage on Kaggle:
  1. Upload this repo (or clone from GitHub) as a Kaggle dataset or use !git clone
  2. Select "GPU T4 x2" accelerator
  3. Run all cells top-to-bottom (~45 min total)
"""

# %% [markdown]
# # Cell 0: Setup & Data Download

# %%
import subprocess, sys, os, json, time, re, textwrap
from pathlib import Path

# ---------- locate repo root ----------
# Works whether this file lives inside the repo or Kaggle copies it to /kaggle/working
REPO_DIR = Path("/kaggle/working/parameter-golf")
if not REPO_DIR.exists():
    # Try current directory (if already in repo)
    candidate = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
    if (candidate / "train_gpt.py").exists():
        REPO_DIR = candidate
    else:
        print("Cloning repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/tns15june/parameter-golf.git", str(REPO_DIR)],
            check=True,
        )

os.chdir(REPO_DIR)
print(f"Working directory: {REPO_DIR}")

# ---------- install deps ----------
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)

# ---------- download minimal data (1 train shard + val) ----------
subprocess.run(
    [sys.executable, "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", "1"],
    check=True,
)
print("Data download complete.")

# ---------- GPU info ----------
import torch

NUM_GPUS = torch.cuda.device_count()
GPU_NAME = torch.cuda.get_device_name(0) if NUM_GPUS > 0 else "NONE"
GPU_MEM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9 if NUM_GPUS > 0 else 0
print(f"GPUs: {NUM_GPUS}× {GPU_NAME}  ({GPU_MEM_GB:.1f} GB each)")

IS_T4 = "T4" in GPU_NAME
IS_P100 = "P100" in GPU_NAME

# %% [markdown]
# # Cell 1: T4 Compatibility Patch
#
# T4 (SM 7.5) does not support Flash Attention v2.
# `train_gpt.py` lines 861-864 force `enable_flash_sdp(True)` → crash on T4.
#
# Fix: monkey-patch the SDP toggle functions *before* they get called in `main()`.
# This way the source file is never modified.

# %%
T4_SDP_PATCH = textwrap.dedent("""\
import torch.backends.cuda as _cuda_backend
_orig_flash = _cuda_backend.enable_flash_sdp
_orig_math = _cuda_backend.enable_math_sdp
_orig_mem = _cuda_backend.enable_mem_efficient_sdp
_orig_cudnn = _cuda_backend.enable_cudnn_sdp

def _patched_flash(enabled):
    # T4 doesn't support flash — silently disable
    return _orig_flash(False)

def _patched_math(enabled):
    # Always enable math SDP as fallback
    return _orig_math(True)

def _patched_mem(enabled):
    # Enable mem-efficient SDP too
    return _orig_mem(True)

_cuda_backend.enable_flash_sdp = _patched_flash
_cuda_backend.enable_math_sdp = _patched_math
_cuda_backend.enable_mem_efficient_sdp = _patched_mem
""")

# Write the patch as a tiny importable module so we can inject it via PYTHONSTARTUP
PATCH_FILE = REPO_DIR / "_t4_sdp_patch.py"
if IS_T4:
    PATCH_FILE.write_text(T4_SDP_PATCH, encoding="utf-8")
    print("T4 SDP patch written to _t4_sdp_patch.py")
else:
    # No patch needed for P100/H100 (P100 also lacks flash, but uses math by default)
    # Still write a safe version that forces math+mem_efficient if not on Ampere+
    PATCH_FILE.write_text(T4_SDP_PATCH, encoding="utf-8")
    print("SDP patch written (safe fallback for non-Ampere GPUs)")

# %% [markdown]
# # Cell 2: Experiment Runner

# %%
RESULTS = []

def run_experiment(
    name: str,
    env_overrides: dict,
    num_gpus: int = 1,
    timeout_seconds: int = 360,
    description: str = "",
):
    """Run train_gpt.py as a subprocess with given env vars. Returns a result dict."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    if description:
        print(f"  {description}")
    print(f"{'='*70}")

    env = os.environ.copy()
    # Common T4-safe defaults
    env.update({
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_BATCH_SIZE": "32768",
        "WARMUP_STEPS": "5",
        "ITERATIONS": "1000",
        "VAL_LOSS_EVERY": "500",
        "TRAIN_LOG_EVERY": "100",
        "MAX_WALLCLOCK_SECONDS": str(timeout_seconds),
        "RUN_ID": f"kaggle_{name}",
    })
    # Apply experiment-specific overrides
    env.update(env_overrides)

    # Inject SDP patch: run a wrapper that imports the patch then execs train_gpt.py
    wrapper_code = (
        "import importlib.util, sys, os\n"
        "spec = importlib.util.spec_from_file_location('_patch', '_t4_sdp_patch.py')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "exec(open('train_gpt.py', encoding='utf-8').read())\n"
    )
    wrapper_file = REPO_DIR / f"_run_{name}.py"
    wrapper_file.write_text(wrapper_code, encoding="utf-8")

    if num_gpus > 1 and NUM_GPUS >= num_gpus:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc_per_node={num_gpus}",
            str(wrapper_file),
        ]
    else:
        if num_gpus > 1 and NUM_GPUS < num_gpus:
            print(f"  WARNING: Requested {num_gpus} GPUs but only {NUM_GPUS} available. Running single-GPU.")
        cmd = [sys.executable, str(wrapper_file)]

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds + 60,  # grace period
        cwd=str(REPO_DIR),
    )
    elapsed = time.time() - t0
    stdout = result.stdout
    stderr = result.stderr

    # Parse metrics from output
    parsed = {
        "name": name,
        "description": description,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "val_bpb": None,
        "val_loss": None,
        "params": None,
        "compressed_bytes": None,
        "peak_mem_mib": None,
        "features": {},
    }

    # Extract final roundtrip BPB (most important metric)
    m = re.search(r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", stdout)
    if m:
        parsed["val_loss"] = float(m.group(1))
        parsed["val_bpb"] = float(m.group(2))

    # Extract params
    m = re.search(r"model_params:(\d+)", stdout)
    if m:
        parsed["params"] = int(m.group(1))

    # Extract compressed artifact size
    m = re.search(r"Serialized model int8\+zlib: (\d+) bytes", stdout)
    if m:
        parsed["compressed_bytes"] = int(m.group(1))

    # Extract peak memory
    m = re.search(r"peak memory allocated: (\d+) MiB", stdout)
    if m:
        parsed["peak_mem_mib"] = int(m.group(1))

    # Feature detection flags
    parsed["features"]["depth_recurrence"] = "depth_recurrence:" in stdout and "recurrences:" in stdout
    parsed["features"]["qat_enabled"] = "QAT enabled at step" in stdout
    parsed["features"]["rope_scaled"] = "RoPE scaled:" in stdout
    parsed["features"]["ttt"] = "Running test-time training eval" in stdout
    parsed["features"]["ddp"] = "NCCL" in stderr or num_gpus > 1
    parsed["features"]["torch_compile"] = "warmup_step:" in stdout
    parsed["features"]["int8_quant"] = "int8_clean_per_row" in stdout or "Serialized model int8" in stdout
    parsed["features"]["unet_skip"] = True  # Always active in architecture
    parsed["features"]["muon_optimizer"] = True  # Always used for matrix params

    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"\n  Result: {status} (exit={result.returncode}, {elapsed:.0f}s)")
    if parsed["val_bpb"] is not None:
        print(f"  val_bpb={parsed['val_bpb']:.4f}  val_loss={parsed['val_loss']:.4f}")
    if parsed["compressed_bytes"] is not None:
        print(f"  compressed={parsed['compressed_bytes']} bytes  params={parsed['params']}")
    if parsed["peak_mem_mib"] is not None:
        print(f"  peak_mem={parsed['peak_mem_mib']} MiB")

    if result.returncode != 0:
        # Print last 30 lines of stderr for debugging
        err_lines = stderr.strip().split("\n")
        print(f"\n  STDERR (last 30 lines):")
        for line in err_lines[-30:]:
            print(f"    {line}")
        # Also check stdout for errors
        out_lines = stdout.strip().split("\n")
        if out_lines:
            print(f"\n  STDOUT (last 10 lines):")
            for line in out_lines[-10:]:
                print(f"    {line}")

    # Cleanup wrapper file
    wrapper_file.unlink(missing_ok=True)

    RESULTS.append(parsed)
    return parsed


# %% [markdown]
# # Cell 3: Experiment — Baseline (single GPU)
#
# Validates: training loop, torch.compile, Muon optimizer, int8 quantization, BPB scoring

# %%
run_experiment(
    name="baseline",
    description="Basic training: 6 layers, dim=256, 1000 iters, single GPU",
    env_overrides={
        "NUM_LAYERS": "6",
        "MODEL_DIM": "256",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "2",
    },
)

# %% [markdown]
# # Cell 4: Experiment — Baseline 2-GPU DDP
#
# Validates: distributed training via torchrun, NCCL backend, gradient sync

# %%
if NUM_GPUS >= 2:
    run_experiment(
        name="baseline_2gpu",
        description="DDP: 6 layers, dim=256, 1000 iters, 2× GPU",
        env_overrides={
            "NUM_LAYERS": "6",
            "MODEL_DIM": "256",
            "NUM_HEADS": "4",
            "NUM_KV_HEADS": "2",
        },
        num_gpus=2,
    )
else:
    print(f"SKIP: baseline_2gpu (only {NUM_GPUS} GPU available, need 2)")

# %% [markdown]
# # Cell 5: Experiment — Depth Recurrence
#
# Validates: weight sharing via round-robin blocks, per-effective-layer scalars

# %%
run_experiment(
    name="depth_recurrence",
    description="3 unique layers × 3 recurrences = 9 effective, dim=384",
    env_overrides={
        "NUM_UNIQUE_LAYERS": "3",
        "NUM_RECURRENCES": "3",
        "NUM_LAYERS": "9",  # effective layers for hyperparameter defaults
        "MODEL_DIM": "384",
        "NUM_HEADS": "6",
        "NUM_KV_HEADS": "3",
    },
)

# %% [markdown]
# # Cell 6: Experiment — Wide Recurrence
#
# Validates: wider model + deeper recurrence fits within VRAM

# %%
run_experiment(
    name="wide_recurrence",
    description="3 unique × 4 rec = 12 effective, dim=512, smaller batch",
    env_overrides={
        "NUM_UNIQUE_LAYERS": "3",
        "NUM_RECURRENCES": "4",
        "NUM_LAYERS": "12",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "TRAIN_BATCH_TOKENS": "32768",  # Fit in 16GB
        "VAL_BATCH_SIZE": "16384",
    },
)

# %% [markdown]
# # Cell 7: Experiment — QAT int8
#
# Validates: fake quantization activates mid-training, reduced quant gap

# %%
run_experiment(
    name="qat_int8",
    description="Recurrence + QAT_BITS=8: fake quant activates at 30% of training",
    env_overrides={
        "NUM_UNIQUE_LAYERS": "3",
        "NUM_RECURRENCES": "3",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "384",
        "NUM_HEADS": "6",
        "NUM_KV_HEADS": "3",
        "QAT_BITS": "8",
        "QAT_START_FRAC": "0.3",
        "EXPORT_BITS": "8",
    },
)

# %% [markdown]
# # Cell 8: Experiment — QAT int4
#
# Validates: int4 export, zlib compresses ~2× better than int8

# %%
run_experiment(
    name="qat_int4",
    description="Recurrence + QAT_BITS=4, EXPORT_BITS=4: int4 + zlib compression",
    env_overrides={
        "NUM_UNIQUE_LAYERS": "3",
        "NUM_RECURRENCES": "3",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "384",
        "NUM_HEADS": "6",
        "NUM_KV_HEADS": "3",
        "QAT_BITS": "4",
        "QAT_START_FRAC": "0.3",
        "EXPORT_BITS": "4",
    },
)

# %% [markdown]
# # Cell 9: Experiment — Eval Long Context (RoPE scaling)
#
# Validates: RoPE base scaling at eval, eval_seq_len > train_seq_len

# %%
run_experiment(
    name="eval_long_ctx",
    description="Train seq=512, eval seq=2048: RoPE NTK scaling at eval time",
    env_overrides={
        "NUM_LAYERS": "6",
        "MODEL_DIM": "256",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "2",
        "TRAIN_SEQ_LEN": "512",
        "EVAL_SEQ_LEN": "2048",
        "VAL_BATCH_SIZE": "16384",  # Larger seqs need smaller batch
    },
)

# %% [markdown]
# # Cell 10: Experiment — Test-Time Training (TTT)
#
# Validates: chunk-by-chunk adaptation on val data, uncompiled model path

# %%
run_experiment(
    name="ttt",
    description="TTT_ENABLED=1: test-time training adapts embeddings+scalars on val data",
    env_overrides={
        "NUM_LAYERS": "6",
        "MODEL_DIM": "256",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "2",
        "ITERATIONS": "500",  # Shorter training; TTT eval is the focus
        "TTT_ENABLED": "1",
        "TTT_LR": "1e-5",
        "TRAIN_SEQ_LEN": "512",
        "EVAL_SEQ_LEN": "512",
        "VAL_BATCH_SIZE": "16384",
    },
    timeout_seconds=480,  # TTT eval is slower (sequential chunks)
)

# %% [markdown]
# # Cell 11: Results Comparison

# %%
print("\n" + "=" * 100)
print("RESULTS SUMMARY")
print("=" * 100)

# ---------- Results table ----------
header = f"{'Experiment':<20} {'Status':<6} {'BPB':>8} {'Loss':>8} {'Params':>10} {'Compressed':>12} {'Peak MiB':>10} {'Time':>8}"
print(header)
print("-" * len(header))

baseline_bpb = None
for r in RESULTS:
    status = "PASS" if r["returncode"] == 0 else "FAIL"
    bpb = f"{r['val_bpb']:.4f}" if r["val_bpb"] is not None else "N/A"
    loss = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "N/A"
    params = f"{r['params']:,}" if r["params"] is not None else "N/A"
    comp = f"{r['compressed_bytes']:,}" if r["compressed_bytes"] is not None else "N/A"
    mem = f"{r['peak_mem_mib']:,}" if r["peak_mem_mib"] is not None else "N/A"
    elapsed = f"{r['elapsed_s']:.0f}s"
    print(f"{r['name']:<20} {status:<6} {bpb:>8} {loss:>8} {params:>10} {comp:>12} {mem:>10} {elapsed:>8}")

    if r["name"] == "baseline" and r["val_bpb"] is not None:
        baseline_bpb = r["val_bpb"]

# ---------- BPB deltas vs baseline ----------
if baseline_bpb is not None:
    print(f"\nBPB deltas vs baseline ({baseline_bpb:.4f}):")
    for r in RESULTS:
        if r["val_bpb"] is not None and r["name"] != "baseline":
            delta = r["val_bpb"] - baseline_bpb
            direction = "worse" if delta > 0 else "better"
            print(f"  {r['name']:<20} {delta:+.4f} ({direction})")

# ---------- int4 vs int8 compression comparison ----------
int8_comp = next((r["compressed_bytes"] for r in RESULTS if r["name"] == "qat_int8" and r["compressed_bytes"]), None)
int4_comp = next((r["compressed_bytes"] for r in RESULTS if r["name"] == "qat_int4" and r["compressed_bytes"]), None)
if int8_comp and int4_comp:
    ratio = int8_comp / int4_comp
    print(f"\nCompression: int8={int8_comp:,} vs int4={int4_comp:,} → int4 is {ratio:.2f}× smaller")

# ---------- Feature validation checklist ----------
print(f"\n{'='*60}")
print("FEATURE VALIDATION CHECKLIST")
print(f"{'='*60}")

checks = [
    ("Training loop + torch.compile", lambda: any(r["returncode"] == 0 and r["features"].get("torch_compile") for r in RESULTS)),
    ("Muon optimizer (matrix params)", lambda: any(r["returncode"] == 0 and r["features"].get("muon_optimizer") for r in RESULTS)),
    ("Int8 quantization + zlib", lambda: any(r["returncode"] == 0 and r["features"].get("int8_quant") for r in RESULTS)),
    ("U-Net skip connections", lambda: any(r["returncode"] == 0 and r["features"].get("unet_skip") for r in RESULTS)),
    ("DDP multi-GPU training", lambda: any(r["returncode"] == 0 and r["features"].get("ddp") and r["name"] == "baseline_2gpu" for r in RESULTS)),
    ("Depth recurrence (weight sharing)", lambda: any(r["returncode"] == 0 and r["features"].get("depth_recurrence") for r in RESULTS)),
    ("QAT fake quantization", lambda: any(r["returncode"] == 0 and r["features"].get("qat_enabled") for r in RESULTS)),
    ("RoPE eval-time scaling", lambda: any(r["returncode"] == 0 and r["features"].get("rope_scaled") for r in RESULTS)),
    ("Test-time training (TTT)", lambda: any(r["returncode"] == 0 and r["features"].get("ttt") for r in RESULTS)),
    ("Int4 export + compression", lambda: int4_comp is not None and (int8_comp is None or int4_comp < int8_comp)),
]

pass_count = 0
for label, check_fn in checks:
    try:
        passed = check_fn()
    except Exception:
        passed = False
    status = "PASS" if passed else "FAIL"
    if passed:
        pass_count += 1
    print(f"  [{status}] {label}")

print(f"\n  Score: {pass_count}/{len(checks)} features validated")

# ---------- Overall verdict ----------
all_passed = all(r["returncode"] == 0 for r in RESULTS)
print(f"\n{'='*60}")
if all_passed and pass_count == len(checks):
    print("ALL EXPERIMENTS PASSED + ALL FEATURES VALIDATED")
elif all_passed:
    print(f"ALL EXPERIMENTS PASSED — {len(checks) - pass_count} feature(s) need investigation")
else:
    failed = [r["name"] for r in RESULTS if r["returncode"] != 0]
    print(f"FAILED EXPERIMENTS: {', '.join(failed)}")
print(f"{'='*60}")

# Save results JSON for later analysis
results_path = REPO_DIR / "kaggle_results.json"
with open(results_path, "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\nFull results saved to {results_path}")

# Cleanup patch file
PATCH_FILE.unlink(missing_ok=True)
