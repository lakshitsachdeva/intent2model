# Intent2Model - Quick Start Guide

## 🚀 Quick Start

### Option 1: Use the startup script (Easiest)
```bash
chmod +x start.sh
./start.sh
```

### Option 2: Manual start

**Terminal 1 - Backend:**
```bash
cd backend
export GEMINI_API_KEY=AIzaSyDc6lDoHJmM1_YEP4XPdl17349eKvg0JAE
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open **http://localhost:3000** in your browser.

---

## 📖 How to Use

### Step 1: Upload CSV
- Drag & drop your CSV file, or click "choose file"
- The system will analyze it automatically

### Step 2: Train a Model
Just chat naturally:
- **"variety"** → trains model to predict the "variety" column
- **"train model"** → auto-selects first column
- Or tell it any column name

### Step 3: View Results
- **"report"** → shows beautiful charts and metrics
- **"show me results"** → displays model performance

### Step 4: Make Predictions
- **"can you predict for me?"** → starts prediction flow
- **"what would be the prediction if..."** → makes a prediction
- Provide values like: **"sepal.length: 5.1, sepal.width: 3.5, petal.length: 1.4, petal.width: 0.2"**

---

## 💬 Example Conversation

```
You: [uploads iris.csv]
AI: analyzed your dataset
AI: 150 rows • 5 columns
AI: suggested targets: variety, sepal.length, sepal.width
AI: which column should i predict?

You: variety
AI: training model to predict "variety"...
AI: model trained successfully
AI: [shows charts]
AI: accuracy: 1.000
AI: top features: petal.length, petal.width, sepal.length

You: report
AI: [shows detailed charts and metrics]

You: can you predict for me?
AI: sure! i need these features: sepal.length, sepal.width, petal.length, petal.width

You: sepal.length: 5.1, sepal.width: 3.5, petal.length: 1.4, petal.width: 0.2
AI: prediction: Setosa
AI: probabilities:
     Setosa: 99.8%
     Versicolor: 0.2%
     Virginica: 0.0%
```

---

## 🎨 Features

- 📊 **Beautiful Charts**: Metrics, feature importance, CV scores
- 🎨 **Extravagant UI**: Gradient colors, smooth animations
- 🤖 **LLM-Powered**: Gemini AI generates optimal pipelines
- 🔮 **Smart Predictions**: Chat-based prediction interface
- 📈 **Model Comparison**: Tries multiple models, picks best
- ⚡ **Auto-Detection**: Automatically detects task type and metrics

---

## 🛑 Stop Services

```bash
./stop.sh
```

Or press `Ctrl+C` in the terminal where you ran `./start.sh`

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

**Backend not starting?**
- Check if port 8000 is free: `lsof -i :8000`
- Check logs: `tail -f backend.log`

**Frontend not starting?**
- Check if port 3000 is free: `lsof -i :3000`
- Check logs: `tail -f frontend.log`

**Training errors?**
- Make sure your CSV has valid data
- Check that the target column exists
- Try a different column name

---

## 🎯 That's It!

Just run `./start.sh` and start chatting! 🚀
