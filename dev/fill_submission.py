#!/usr/bin/env python3
"""Parse a train.log and fill submission.json with actual metrics.

Usage:
    python3 dev/fill_submission.py logs/submission_v1.txt
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SUBMISSION_DIR = Path("records/track_10min_16mb/2026-04_tns15june_v1")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 dev/fill_submission.py <log_file>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    log_text = log_path.read_text(encoding="utf-8")

    # Parse final roundtrip metrics
    rt = re.search(r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)", log_text)
    if not rt:
        print("ERROR: Could not find final_int8_zlib_roundtrip_exact in log")
        sys.exit(1)

    val_loss = float(rt.group(1))
    val_bpb = float(rt.group(2))

    # Parse sizes (handles both old "int8+zlib:" and new format)
    sz = re.search(r"Total submission size.*?: (\d+) bytes", log_text)
    code_sz = re.search(r"Code size: (\d+) bytes", log_text)

    bytes_total = int(sz.group(1)) if sz else 0
    bytes_code = int(code_sz.group(1)) if code_sz else 0

    submission = {
        "author": "Tarkeshwar Narayan Sharma",
        "github_id": "tns15june",
        "name": "10L/512d LeakyReLU\u00b2 + int6 QAT + EMA + Sliding Window",
        "blurb": (
            "10-layer dim=512 with 2x MLP and LeakyReLU\u00b2 activation. "
            "int6 QAT (start at 15%) + int8 embeddings, LZMA-compressed. "
            "EMA(0.9995) of weights for export. Sliding-window eval (seq_len=1024, stride=256). "
            f"Post-quant roundtrip BPB: {val_bpb:.4f}."
        ),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "bytes_total": bytes_total,
        "bytes_code": bytes_code,
    }

    out_path = SUBMISSION_DIR / "submission.json"
    out_path.write_text(json.dumps(submission, indent=2) + "\n", encoding="utf-8")
    print(f"Written: {out_path}")
    print(f"  val_bpb: {val_bpb:.8f}")
    print(f"  bytes_total: {bytes_total}")
    print(f"  beats baseline by: {1.2244 - val_bpb:.4f} BPB")

    # Also copy the log
    log_dest = SUBMISSION_DIR / "train.log"
    log_dest.write_text(log_text, encoding="utf-8")
    print(f"  Copied log to: {log_dest}")


if __name__ == "__main__":
    main()
