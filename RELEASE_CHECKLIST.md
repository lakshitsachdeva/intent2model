# Drift — Release checklist

This file is for **you (the human)** preparing a public release. It does not run or publish anything.

---

## (A) What is prepared in code

- **Engine build:** `.github/workflows/build-engine.yml` — on push of tag `v*`, builds the drift engine binary for macOS (arm64), Linux (x64), Windows (x64), and creates a **draft** GitHub Release with those binaries attached. It does **not** publish the release.
- **npm package:** `drift-npm/` — provides the global `drift` command; on first run it detects OS and architecture, downloads the correct engine binary into `~/.drift/bin/`, starts the engine in the background, then runs the chat CLI. Engine source is never shipped.
- **User docs:** `drift-npm/README.md` and the section below — “Exactly what to do” for users; engine runs locally; LLM required; web UI can be on Vercel; no mention of backend, FastAPI, or Python in user-facing copy.

---

## (B) Copy-paste: “Exactly what to do” (for users)

You can use this in your docs, landing page, or npm package description:

```markdown
## Exactly what to do

1. **Install drift**
   ```bash
   npm install -g drift-ml
   ```
   (Requires Node.js ≥ 18 and Python 3 on your PATH.)

2. **Run drift**
   ```bash
   drift
   ```
   On first run, the CLI downloads and starts the drift engine locally. You’ll see a short welcome and instructions in the terminal.

3. **Use an LLM**
   Training and planning use a local LLM. You need one of:
   - **Gemini CLI** — install and set `GEMINI_API_KEY` or have `gemini` on your PATH.
   - **Ollama** — run `ollama run llama2` (or another model).
   - Another local LLM the engine supports.

4. **Chat**
   - `load path/to/data.csv`
   - `predict price` (or any column)
   - `try something stronger`
   - `why is accuracy capped`
   - `quit` to exit.

The engine runs on your machine. The web app (if you use it) can be hosted on Vercel; the engine stays local.
```

---

## (C) Your manual steps (in order)

Do these yourself. Nothing below is executed or published by code.

1. **Tag and push to trigger the engine build**
   - Create and push a tag, e.g. `v0.1.0`:
     ```bash
     git tag v0.1.0
     git push origin v0.1.0
     ```
   - Wait for the GitHub Actions workflow “Build engine and draft release” to finish.

2. **Publish the draft release**
   - On GitHub: **Releases** → open the new **draft** release for that tag.
   - Review the attached binaries (e.g. `drift-engine-macos-arm64`, `drift-engine-linux-x64`, `drift-engine-windows-x64.exe`).
   - Click **Publish release**.

3. **npm: log in**
   - Run `npm login` and complete the prompts (username, password, OTP if enabled).

4. **Publish the npm package**
   - From the **project root** (or from `drift-npm/` if the package lives there):
     ```bash
     cd drift-npm
     npm publish --access public
     ```
   - Use `--access public` if the package is scoped or you want it public.

5. **Point CLI at your release URL (if repo is not “drift”)**
   - The CLI downloads the engine from `DRIFT_ENGINE_BASE_URL` (default in code: `https://github.com/lakshitsachdeva/drift/releases/latest/download`).
   - If you build and publish releases from this repo (e.g. `intent2model`), tell users to set `DRIFT_ENGINE_BASE_URL=https://github.com/YOUR_USER/YOUR_REPO/releases/latest/download` or update the default in `drift-npm/bin/drift.js` and republish the npm package.

6. **Optional: deploy the web UI to Vercel**
   - Connect your repo to Vercel and deploy the **frontend** (e.g. `frontend/` or the root with the correct build settings).
   - No backend or engine is deployed; users run the engine locally.

---

## (D) What was not done (and you must not assume)

- No tag was pushed.
- No GitHub Release was published.
- No `npm login` or `npm publish` was run.
- No Vercel (or other) deployment was performed.
- No execution or filesystem access was claimed; only files were created or edited to prepare the above.
