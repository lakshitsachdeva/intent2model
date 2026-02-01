# Drift engine binary build

Builds the **drift engine** as a single binary (closed distribution). Same HTTP API as the web app; no source shipped.

## Prerequisites

- Python 3.10+ with project deps: `pip install -r requirements.txt` then `pip install pyinstaller`
- Run from **project root** (parent of `backend/` and `engine_build/`)

## Build (one binary per OS)

From project root:

```bash
pyinstaller engine_build/drift-engine.spec
```

Output:

- **macOS:** `dist/drift-engine-macos-arm64` or `dist/drift-engine-macos-x64`
- **Linux:** `dist/drift-engine-linux-x64` or `dist/drift-engine-linux-arm64`
- **Windows:** `dist/drift-engine-windows-x64.exe`

Rename or upload these for the CLI to download (see drift-npm: `DRIFT_ENGINE_BASE_URL` + asset names).

## Run the binary locally

```bash
./dist/drift-engine-macos-arm64   # or your platform
# Listens on http://0.0.0.0:8000 by default. Override with DRIFT_ENGINE_PORT.
```

## Release

Upload each platform binary to your release URL so the drift CLI can fetch them on first run (e.g. GitHub Releases, then set `DRIFT_ENGINE_BASE_URL`).
