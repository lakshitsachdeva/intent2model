# drift-ml

**Drift** by Lakshit Sachdeva — terminal-first, chat-based AutoML. Same engine as the web app. Local-first: the engine runs on your machine; no commands to memorize.

---

## Exactly what to do

1. **Install drift**
   ```bash
   npm install -g drift-ml
   ```
   Requires Node.js ≥ 18 and Python 3 on your PATH.

2. **Run drift**
   ```bash
   drift
   ```
   On first run, the CLI detects your OS and architecture, downloads the correct engine binary into `~/.drift/bin/`, and starts the engine in the background. You’ll see a short welcome and instructions in the terminal.

3. **Use a local LLM**
   Training and planning use an LLM. You need one of:
   - **Gemini CLI** — install it and set `GEMINI_API_KEY` or have `gemini` on your PATH.
   - **Ollama** — run `ollama run llama2` (or another model).
   - Another local LLM the engine supports.

4. **Chat**
   - `load path/to/data.csv`
   - `predict price` (or any column)
   - `try something stronger`
   - `why is accuracy capped`
   - `quit` to exit.

That’s it. The engine runs locally. The web app (if you use it) can be hosted on Vercel; the engine stays on your machine.

---

## What is drift?

- **Local-first** — The engine runs on your machine. Training and planning stay local; you never send data to our servers.
- **Terminal-first, chat-based** — Same engine as the web app. No commands to memorize; chat in natural language.
- **Engine** — On first run the CLI downloads and starts the engine from `~/.drift/bin/`. Or set `DRIFT_BACKEND_URL` to a running engine URL.

---

## Install (details)

```bash
npm install -g drift-ml
drift
```

The `drift` command installs or upgrades the chat CLI (Python) and runs it. You get the welcome and instructions every time.

### Alternative: pipx (Python only — macOS, Linux, Windows)

```bash
pipx install drift-ml
drift
```

**Update (pipx):**
```bash
pipx upgrade drift-ml
```
(PowerShell on Windows: same command.)

---

## Example usage

```text
drift › load iris.csv
drift › predict sepal.length
drift › try something stronger
drift › why is accuracy capped
drift › quit
```

---

## License

MIT
