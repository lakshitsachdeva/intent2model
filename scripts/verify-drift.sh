#!/usr/bin/env bash
# Verify drift engine releases and npm/pipx setup.
# Run from repo root.
set -e
REPO="lakshitsachdeva/drift"
API="https://api.github.com/repos/${REPO}/releases"

echo "=== Drift release check ==="
echo "Repo: $REPO"
echo ""

# Check releases
RELEASES=$(curl -s "$API")
if ! echo "$RELEASES" | grep -q '"tag_name"'; then
  echo "❌ NO RELEASES in $REPO"
  echo ""
  echo "Fix:"
  echo "  1. Add DRIFT_RELEASES_TOKEN to intent2model → Settings → Secrets → Actions"
  echo "     (PAT with repo scope for $REPO)"
  echo "  2. Push a tag to trigger the workflow:"
  echo "     git tag v0.1.4 && git push origin v0.1.4"
  echo "  3. Wait for workflow: https://github.com/lakshitsachdeva/intent2model/actions"
  echo "  4. Re-run this script"
  exit 1
fi

# Parse latest tag
TAGS=$(echo "$RELEASES" | grep -o '"tag_name": "[^"]*"' | head -5)
echo "✓ Releases found:"
echo "$TAGS" | sed 's/"tag_name": "/  - /;s/"$//'
echo ""

# Check what npm expects
NPM_TAG=$(grep -r "ENGINE_TAG" drift-npm/bin/drift.js 2>/dev/null | grep -o 'v[0-9.]*' | head -1)
if [[ -n "$NPM_TAG" ]]; then
  if echo "$RELEASES" | grep -q "\"tag_name\": \"$NPM_TAG\""; then
    echo "✓ npm/pipx expect $NPM_TAG — release exists"
  else
    echo "⚠ npm/pipx expect $NPM_TAG — NOT in releases"
    echo "  Update ENGINE_TAG in drift-npm/bin/drift.js and drift/engine_launcher.py to match a release"
  fi
fi
echo ""
echo "=== pipx check ==="
if command -v drift &>/dev/null; then
  echo "✓ drift command found: $(which drift)"
else
  echo "❌ drift not in PATH. Run: pipx install drift-ml"
fi
echo ""
echo "=== npm check ==="
if command -v npm &>/dev/null; then
  NPM_DRIFT=$(npm list -g drift-ml 2>/dev/null || true)
  if echo "$NPM_DRIFT" | grep -q drift-ml; then
    echo "✓ drift-ml installed: $NPM_DRIFT"
  else
    echo "  npm install -g drift-ml (optional, pipx is enough)"
  fi
fi
