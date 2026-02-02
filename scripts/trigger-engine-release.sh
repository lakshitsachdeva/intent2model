#!/usr/bin/env bash
# Trigger engine build + release to drift by pushing a tag.
# Prerequisite: DRIFT_RELEASES_TOKEN in intent2model → Settings → Secrets → Actions
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="${1:-v0.1.4}"
echo "Will create and push tag: $TAG"
echo "This triggers the workflow to build and publish to lakshitsachdeva/drift"
echo ""

# Update ENGINE_TAG in code to match
echo "Updating ENGINE_TAG to $TAG..."
sed -i.bak "s/ENGINE_TAG = \"v[0-9.]*\"/ENGINE_TAG = \"$TAG\"/" drift/engine_launcher.py
sed -i.bak "s/ENGINE_TAG = \"v[0-9.]*\"/ENGINE_TAG = \"$TAG\"/" drift-npm/bin/drift.js
rm -f drift/engine_launcher.py.bak drift-npm/bin/drift.js.bak

git add drift/engine_launcher.py drift-npm/bin/drift.js
git diff --cached --quiet && echo "No changes" || git commit -m "chore: pin engine to $TAG"
git push origin main

# Create and push tag
git tag -f "$TAG" && git push origin "$TAG"

echo ""
echo "Done. Check workflow: https://github.com/lakshitsachdeva/intent2model/actions"
echo "When green, run: bash scripts/verify-drift.sh"
