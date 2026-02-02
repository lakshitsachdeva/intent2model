#!/usr/bin/env bash
# Publish drift-ml to npm. Run from repo root.
# If "token expired": run `npm login` first.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/drift-npm"
npm publish --access public
