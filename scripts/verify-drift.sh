#!/usr/bin/env bash
# Verify drift engine releases. Run from repo root.
set -e
REPO="lakshitsachdeva/intent2model"
API="https://api.github.com/repos/${REPO}/releases"

echo "=== Drift release check ==="
echo "Repo: $REPO"
echo ""

RELEASES=$(curl -s "$API")
if ! echo "$RELEASES" | grep -q '"tag_name"'; then
  echo "❌ No releases. Push a tag: git tag v0.2.0 && git push origin v0.2.0"
  exit 1
fi

echo "✓ Releases:"
echo "$RELEASES" | grep -o '"tag_name": "[^"]*"' | head -5 | sed 's/"tag_name": "/  - /;s/"$//'
echo ""
echo "drift uses latest release automatically. No tokens required."
