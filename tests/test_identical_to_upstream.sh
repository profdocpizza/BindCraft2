#!/usr/bin/env bash
# With no charge budget set, decoding must be byte-identical to the base revision.
# Usage: bash tests/test_identical_to_upstream.sh [revision]   (default: upstream/main)
set -euo pipefail
root=$(git rev-parse --show-toplevel)
revision=${1:-upstream/main}
scratch=$(mktemp -d)
trap 'git -C "$root" worktree remove --force "$scratch/base" >/dev/null 2>&1 || true; rm -rf "$scratch"' EXIT

git -C "$root" worktree add --detach "$scratch/base" "$revision" >/dev/null
mkdir -p "$scratch/base/tests"
cp "$root/tests/dump_sequences.py" "$scratch/base/tests/dump_sequences.py"

JAX_PLATFORMS=cpu python "$scratch/base/tests/dump_sequences.py" > "$scratch/base.tsv"
JAX_PLATFORMS=cpu python "$root/tests/dump_sequences.py" > "$scratch/working.tsv"

if diff -q "$scratch/base.tsv" "$scratch/working.tsv" >/dev/null; then
  echo "PASS identical to $revision ($(wc -l < "$scratch/base.tsv") sequences)"
else
  echo "FAIL output differs from $revision"
  diff "$scratch/base.tsv" "$scratch/working.tsv" | head -20
  exit 1
fi
