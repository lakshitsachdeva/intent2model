#!/bin/bash
# Run backend from backend/ directory. Use from project root: ./backend/run.sh

set -e
cd "$(dirname "$0")"

echo "📦 Starting backend (must run from backend/ directory)..."
echo "   Working directory: $(pwd)"
echo ""

# Port already in use?
if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is in use. Free it with:"
    echo "   lsof -ti:8000 | xargs kill -9"
    echo ""
    read -p "Kill process on 8000 and start anyway? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        exit 1
    fi
fi

# Optional venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run (no reload = simpler; use --reload for dev)
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
