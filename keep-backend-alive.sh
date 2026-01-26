#!/bin/bash

# Keep Backend Alive Script
# This ensures the backend stays running

cd "$(dirname "$0")/backend"

export GEMINI_API_KEY=AIzaSyDc6lDoHJmM1_YEP4XPdl17349eKvg0JAE

echo "🔄 Starting backend (will auto-restart if it crashes)..."
echo "📝 Logs: /tmp/backend.log"
echo "🛑 Press Ctrl+C to stop"
echo ""

while true; do
    echo "🚀 Starting backend..."
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload >> /tmp/backend.log 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Backend crashed, restarting in 3 seconds..."
        sleep 3
    else
        echo "✅ Backend stopped normally"
        break
    fi
done
