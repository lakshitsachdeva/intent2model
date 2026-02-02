# drift

**Terminal-first, chat-based AutoML.** Open source. No tokens. No auth.

---

## Install

```bash
pipx install drift-ml
```

Or:

```bash
npm install -g drift-ml
```

(Both require `pipx install drift-ml` for the Python CLI.)

---

## Run

```bash
drift
```

On first run, drift downloads and starts the engine automatically. No backend setup. No config. No tokens.

---

## How it works

- **Local-first** — Engine runs on your machine. Data never leaves.
- **Chat-based** — `load data.csv`, `predict price`, `try something stronger`
- **Auto-start** — Engine downloads and starts in the background. You never start a backend manually.
- **No auth** — No API keys for drift. (You need an LLM for training: Ollama, Gemini CLI, etc.)

---

## Example

```text
drift › load iris.csv
drift › predict variety
drift › try something stronger
drift › quit
```

---

## Philosophy

drift should feel like `git`, `docker`, `brew` — a tool you trust immediately. Zero friction. Open source.

---

## Web UI (optional)

This repo also includes a web UI. For development:

```bash
./start.sh
```

Then open http://localhost:3000

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
