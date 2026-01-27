# 🔄 How to Restart Intent2Model

## Quick Restart (Easiest Way)

```bash
# Stop everything
./stop.sh

# Start everything
./start.sh
```

That's it! 🎉

---

## Step-by-Step Guide

### 1. **Stop the Services**

```bash
./stop.sh
```

This will:
- Stop the backend server (port 8000)
- Stop the frontend server (port 3000)
- Clean up all processes

**What you'll see:**
```
🛑 Stopping Intent2Model services...
✅ All services stopped
```

### 2. **Start the Services**

```bash
./start.sh
```

This will:
- Load API key from `.env` file
- Start backend server
- Start frontend server
- Show you the URLs

**What you'll see:**
```
🚀 Starting Intent2Model...
✅ Loaded API key from .env file
✅ Backend is running on http://localhost:8000
✅ Frontend is running on http://localhost:3000
🎉 Intent2Model is LIVE!
```

### 3. **Verify It's Working**

Open in browser:
- **Frontend:** http://localhost:3000
- **Backend Health:** http://localhost:8000/health

---

## Changing API Key

### Option 1: Edit `.env` file (Recommended)

1. Open `.env` file in the project root
2. Change the `GEMINI_API_KEY` value
3. Restart: `./stop.sh && ./start.sh`

```bash
# Edit .env file
nano .env
# or
code .env

# Change this line:
GEMINI_API_KEY=your_new_api_key_here
```

### Option 2: Use UI

1. Open http://localhost:3000
2. Click "LLM Settings" button (top right)
3. Enter your API key
4. Click "Set API Key"

---

## Troubleshooting

### Services won't stop?

```bash
# Force kill
pkill -9 -f "uvicorn main:app"
pkill -9 -f "next dev"
```

### Port already in use?

```bash
# Check what's using the port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend

# Kill specific process
kill -9 <PID>
```

### API key not working?

1. Check `.env` file exists and has correct key
2. Check backend logs: `tail -f backend.log`
3. Test API key in UI: Click "LLM Settings" → Enter key → "Set API Key"

### Check logs

```bash
# Backend logs
tail -f backend.log

# Frontend logs  
tail -f frontend.log
```

---

## Manual Start (Alternative)

If scripts don't work, start manually:

**Terminal 1 - Backend:**
```bash
cd backend
source .venv/bin/activate
export GEMINI_API_KEY=your_key_here
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `./stop.sh` | Stop all services |
| `./start.sh` | Start all services |
| `./stop.sh && ./start.sh` | Restart everything |
| `tail -f backend.log` | Watch backend logs |
| `tail -f frontend.log` | Watch frontend logs |

---

## Pro Tips 💡

1. **Always use `./stop.sh` before `./start.sh`** - Prevents port conflicts
2. **Check `.env` file** - Make sure API key is there
3. **Watch the logs** - If something fails, logs will tell you why
4. **Use UI for testing** - The "LLM Settings" button is great for quick API key changes
