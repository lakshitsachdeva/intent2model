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
```bash
cd backend
pip install -r ../requirements.txt
export GEMINI_API_KEY=AIzaSyDc6lDoHJmM1_YEP4XPdl17349eKvg0JAE
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

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

## 📚 More Info

See `HOW_TO_USE.md` for detailed instructions and examples.

---

**That's it! Just run `./start.sh` and start chatting! 🎉**
