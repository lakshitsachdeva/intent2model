# Release checklist

## 1. Tag (triggers engine build)

```bash
git tag v0.2.2   # or next version
git push origin v0.2.2
```

Wait for workflow: https://github.com/lakshitsachdeva/intent2model/actions

## 2. Update ENGINE_TAG

In `drift/engine_launcher.py` and `drift-npm/bin/drift.js`, set:
```python
ENGINE_TAG = "v0.2.2"  # match the tag above
```

Engine binaries publish to `lakshitsachdeva/intent2model` (same repo).
**macOS fix:** Build uses `upx=False` so the binary can be ad-hoc signed (avoids Gatekeeper "killed").

## 3. Bump versions

- `pyproject.toml` → version
- `drift-npm/package.json` → version

## 4. Publish PyPI (from repo root)

```bash
cd /Users/lakshitsachdeva/Desktop/Projects/intent2model
python3 -m build --outdir dist
twine upload dist/drift_ml-0.2.2*
# Or: TWINE_USERNAME=__token__ TWINE_PASSWORD='pypi-xxx' python3 -m twine upload dist/drift_ml-0.2.2*
```

## 5. Publish npm

```bash
npm login   # if token expired
cd drift-npm
npm publish --access public
```

## 6. Verify

```bash
cd /Users/lakshitsachdeva/Desktop/Projects/intent2model
bash scripts/verify-drift.sh
```
