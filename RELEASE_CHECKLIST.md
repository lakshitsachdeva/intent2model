# Drift — Release checklist

## Engine build

Push a tag to trigger the workflow:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The workflow builds binaries for macOS (arm64), Linux (x64), Windows (x64) and publishes a release to this repo. No secrets required — uses default GITHUB_TOKEN.

## Publish npm

```bash
cd drift-npm
npm publish --access public
```

## Publish PyPI

```bash
python -m build --outdir dist
twine upload dist/*
```

## Engine download

Users download from this repo's releases. No tokens. No auth.
