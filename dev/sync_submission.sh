#!/bin/bash
# Sync the root train_gpt.py into the records submission folder.
# Run this any time train_gpt.py changes so the submission folder stays in sync.
# Usage: bash dev/sync_submission.sh

set -e

ROOT_SCRIPT="train_gpt.py"
SUBMISSION_DIR="records/track_10min_16mb/2026-04_tns15june_v1"
SUBMISSION_SCRIPT="$SUBMISSION_DIR/train_gpt.py"

if [ ! -f "$ROOT_SCRIPT" ]; then
    echo "ERROR: $ROOT_SCRIPT not found — run from repo root."
    exit 1
fi
if [ ! -d "$SUBMISSION_DIR" ]; then
    echo "ERROR: $SUBMISSION_DIR not found."
    exit 1
fi

if [ -f "$SUBMISSION_SCRIPT" ] && cmp -s "$ROOT_SCRIPT" "$SUBMISSION_SCRIPT"; then
    echo "Already in sync: $SUBMISSION_SCRIPT"
    exit 0
fi

cp "$ROOT_SCRIPT" "$SUBMISSION_SCRIPT"
ROOT_SHA=$(sha256sum "$ROOT_SCRIPT" | awk '{print $1}')
SUB_SHA=$(sha256sum "$SUBMISSION_SCRIPT" | awk '{print $1}')

echo "Synced $ROOT_SCRIPT -> $SUBMISSION_SCRIPT"
echo "  root  sha256: $ROOT_SHA"
echo "  sub   sha256: $SUB_SHA"

if [ "$ROOT_SHA" != "$SUB_SHA" ]; then
    echo "ERROR: checksums differ after copy"
    exit 1
fi
