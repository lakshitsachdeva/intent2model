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

## Example (CLI)

```text
drift › load iris.csv
drift › predict variety
drift › try something stronger
drift › quit
```

---

## Use as library

```bash
pip install drift-ml
```

```python
from drift import Drift

d = Drift()
d.load("iris.csv")
d.chat("predict sepal length")
result = d.train()
print(result["metrics"])
```

Or with an existing engine:

```python
from drift import Drift

d = Drift(base_url="http://localhost:8000")
d.load("data.csv")
reply = d.chat("predict price")
print(d.get_last_reply(reply))
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
