#!/usr/bin/env bash
# Publish drift-ml to npm and PyPI. Run from repo root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Building Python package (drift-ml)"
python3 -m build --outdir dist 2>/dev/null || pip install build && python3 -m build --outdir dist

echo "==> Publishing to PyPI"
VER=$(grep '^version = ' pyproject.toml 2>/dev/null | cut -d'"' -f2)
if [[ -n "$VER" ]] && ls dist/drift_ml-${VER}* 1>/dev/null 2>&1; then
  python3 -m twine upload dist/drift_ml-${VER}* --skip-existing 2>/dev/null || echo "Run: twine upload dist/drift_ml-${VER}*"
else
  echo "Run: twine upload dist/drift_ml-<version>*"
fi

echo "==> Publishing npm (drift-ml)"
cd drift-npm
npm publish --access public 2>/dev/null || echo "Run 'npm login' first, then: npm publish --access public"
cd "$ROOT"

echo "==> Done. pipx install drift-ml  |  npm install -g drift-ml"
