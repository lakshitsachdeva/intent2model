# drift

**Terminal-first, chat-based AutoML.** Open source. No tokens. No auth. Works fully local.

---

## Install

```bash
pipx install drift-ml
```

Or with npm (also requires pipx for the CLI):

```bash
pipx install drift-ml
npm install -g drift-ml
```

---

## Run

```bash
drift
```

That's it. On first run, drift downloads and starts the engine automatically. No backend setup. No config.

---

## How it works

- **Local-first** — Engine runs on your machine. Data never leaves.
- **Chat-based** — `load data.csv`, `predict price`, `try something stronger`
- **Auto-start** — Engine downloads and starts in the background. You never touch it.
- **No tokens** — No API keys for drift. (You need an LLM for training: Ollama, Gemini CLI, etc.)

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

---

## Philosophy

drift should feel like `git`, `docker`, `brew` — a tool you trust immediately. Zero friction. Open source.

---

## License

MIT
