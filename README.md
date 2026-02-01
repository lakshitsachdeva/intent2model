# Intent2Model 🚀

LLM-Guided AutoML Agent - Upload a CSV, chat with the AI, get a trained model.

## ⚡ Quick Start (Easiest Way)

```bash
# Make scripts executable (first time only)
chmod +x start.sh stop.sh

# Start everything
./start.sh
```

Then open **http://localhost:3000** in your browser!

To stop: Press `Ctrl+C` or run `./stop.sh`

---

## 📖 How to Use

### 1. Upload CSV
- Drag & drop your CSV file or click "choose file"
- System analyzes it automatically

### 2. Train Model
Just type a column name in the chat:
- **"variety"** → trains model to predict "variety"
- **"price"** → trains model to predict "price"
- Or any column name from your dataset

### 3. View Results
- **"report"** → shows beautiful charts and metrics
- **"show me results"** → displays model performance

### 4. Make Predictions
- **"predict"** or **"can you predict for me?"** → starts prediction flow
- Provide feature values: **"sepal.length: 5.1, sepal.width: 3.5"**

---

## 💬 Example Conversation

```
You: [uploads iris.csv]
AI: ✓ analyzed your dataset • 150 rows • 5 columns
AI: suggested targets: variety, sepal.length, sepal.width

You: variety
AI: 🚀 training model to predict "variety"...
AI: ✅ model trained successfully!
AI: accuracy: 1.000 • best model: RandomForest
AI: [shows charts]

You: report
AI: [shows detailed charts: metrics, feature importance, CV scores]

You: predict
AI: sure! i need: sepal.length, sepal.width, petal.length, petal.width

You: sepal.length: 5.1, sepal.width: 3.5, petal.length: 1.4, petal.width: 0.2
AI: 🎯 prediction: Setosa
AI: probabilities: Setosa 99.8%, Versicolor 0.2%, Virginica 0.0%
```

---

## 🛠️ Manual Setup (Alternative)

### Backend
**Important:** Always run uvicorn from inside the `backend/` folder. Running from project root will fail with "Could not import module main".

```bash
cd backend
pip install -r ../requirements.txt
# Optional: set API key in .env or export GEMINI_API_KEY=your_key
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the helper script (from project root):
```bash
chmod +x backend/run.sh
./backend/run.sh
```

**Backend nahi chal raha?**
- **"Could not import module main"** → You're in the wrong folder. Run `cd backend` first, then `python3 -m uvicorn main:app --host 0.0.0.0 --port 8000`.
- **"Address already in use" / port 8000** → Free the port: `lsof -ti:8000 | xargs kill -9`, then start again.
- **Dependencies missing** → From project root: `pip install -r requirements.txt` (or use `./start.sh` — it creates a venv and installs deps).

### Frontend (New Terminal)
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:3000**

---

## 🎨 Features

- 📊 **Beautiful Charts**: Metrics, feature importance, CV scores
- 🎨 **Extravagant UI**: Gradient colors, smooth animations
- 🤖 **LLM-Powered**: Gemini AI generates optimal pipelines
- 🔮 **Smart Predictions**: Chat-based prediction interface
- 📈 **Model Comparison**: Tries multiple models, picks best
- ⚡ **Auto-Detection**: Automatically detects task type and metrics

---

## 📝 Requirements

- Python 3.10+
- Node.js 18+
- npm/yarn

Install dependencies:
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

---

## 🐛 Troubleshooting

**Services not starting?**
- Check ports: `lsof -i :8000` and `lsof -i :3000`
- Check logs: `tail -f backend.log` or `tail -f frontend.log`

**Training errors?**
- Make sure CSV has valid data
- Check that target column exists
- Try a different column name

---

## drift — Terminal-first CLI

**drift** by Lakshit Sachdeva. Terminal-first, chat-based AutoML — same engine as the web UI. No commands to memorize.

### Exactly what to do (any computer)

1. **Install drift** (pick one):
   ```bash
   npm install -g drift-ml
   ```
   or:
   ```bash
   pipx install drift
   ```

2. **Run drift:**
   ```bash
   drift
   ```
   You’ll see the welcome and step-by-step instructions in the terminal.

3. **Engine** — On first run the CLI downloads and starts the drift engine locally (or set `DRIFT_BACKEND_URL` to a running engine). You need an LLM: Gemini CLI, Ollama, or another local LLM.

4. **In drift:** type `load path/to/your.csv`, then chat (e.g. `predict price`, `try something stronger`). Type `quit` to exit.

drift shows you the rest when you run it.

### Install (details)

- **Local-first** — Same engine as the web app; planning and training run on your machine.
- **Chat-based**: e.g. `load iris.csv`, `predict price`, `try something stronger`, `why is accuracy capped`.
- **Engine** runs locally (CLI auto-starts it or use `DRIFT_BACKEND_URL`). Web UI can be hosted on Vercel.

---

## 📚 More Info

See `HOW_TO_USE.md` for detailed instructions and examples.

---

**That's it! Just run `./start.sh` and start chatting! 🎉**
