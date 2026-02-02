#!/usr/bin/env bash
# Push README to lakshitsachdeva/drift so the repo isn't empty.
# Run from repo root. Requires: git, push access to drift.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DRIFT_DIR="/tmp/drift-repo-init"
rm -rf "$DRIFT_DIR"
git clone https://github.com/lakshitsachdeva/drift.git "$DRIFT_DIR"
cd "$DRIFT_DIR"
cp "$ROOT/drift-repo/README.md" .
git add README.md
git status
git diff --cached --quiet || git commit -m "Add README"
git push origin main
rm -rf "$DRIFT_DIR"
echo "Done. drift repo now has README."
