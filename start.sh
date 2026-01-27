#!/bin/bash

# Intent2Model Startup Script
# This script starts both backend and frontend services

set +e  # Don't exit on error, we'll handle it manually

echo "🚀 Starting Intent2Model..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "${RED}❌ Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Check for required commands
command -v python3 >/dev/null 2>&1 || { echo "${RED}❌ Error: python3 is required but not installed${NC}"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "${RED}❌ Error: node is required but not installed${NC}"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "${RED}❌ Error: npm is required but not installed${NC}"; exit 1; }

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
sleep 2

# Set API key
export GEMINI_API_KEY=AIzaSyAuxa5b792g6AaiD_ZURSrvGvLh-M-3bUw

# Start Backend
echo ""
echo "${BLUE}📦 Starting Backend...${NC}"
cd backend

# Use a local virtualenv to avoid macOS PEP-668 (externally-managed env) issues
if [ ! -d ".venv" ]; then
    echo "${YELLOW}⚠️  Creating Python virtualenv (backend/.venv)...${NC}"
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Check/install Python dependencies inside venv
if ! python -c "import uvicorn" 2>/dev/null; then
    echo "${YELLOW}⚠️  Installing Python dependencies into venv...${NC}"
    python -m pip install --upgrade pip setuptools wheel --quiet
    python -m pip install -r ../requirements.txt --quiet
    echo "${GREEN}✅ Python dependencies installed${NC}"
fi

# Check/install google-generativeai for LLM support
if ! python -c "import google.generativeai" 2>/dev/null; then
    echo "${YELLOW}⚠️  Installing google-generativeai for LLM support...${NC}"
    python -m pip install google-generativeai --quiet
    echo "${GREEN}✅ LLM dependencies installed${NC}"
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "${RED}❌ Error: main.py not found in backend directory${NC}"
    exit 1
fi

# Start backend in background
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
MAX_RETRIES=10
RETRY_COUNT=0
BACKEND_RUNNING=false
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "${GREEN}✅ Backend is running on http://localhost:8000${NC}"
        BACKEND_RUNNING=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "⏳ Waiting for backend... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ "$BACKEND_RUNNING" = false ]; then
    echo "${RED}❌ Backend failed to start. Check logs: tail -f backend.log${NC}"
    tail -20 backend.log 2>/dev/null || echo "No log file yet"
    exit 1
fi

# Start Frontend
echo ""
echo "${BLUE}🎨 Starting Frontend...${NC}"
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "${YELLOW}⚠️  Installing npm dependencies...${NC}"
    npm install --silent
    echo "${GREEN}✅ npm dependencies installed${NC}"
fi

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "${RED}❌ Error: package.json not found in frontend directory${NC}"
    exit 1
fi

# Start frontend in background
nohup npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 8

# Check if frontend is running
MAX_RETRIES=15
RETRY_COUNT=0
FRONTEND_RUNNING=false
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "${GREEN}✅ Frontend is running on http://localhost:3000${NC}"
        FRONTEND_RUNNING=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "⏳ Waiting for frontend... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    fi
done

if [ "$FRONTEND_RUNNING" = false ]; then
    echo "${YELLOW}⚠️  Frontend might still be starting. Check logs: tail -f frontend.log${NC}"
    tail -20 frontend.log 2>/dev/null || echo "No log file yet"
fi

# Print status
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ${GREEN}🎉 Intent2Model is LIVE!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🌐 Frontend: ${BLUE}http://localhost:3000${NC}"
echo "  🔧 Backend:  ${BLUE}http://localhost:8000${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "${YELLOW}📖 How to Use:${NC}"
echo ""
echo "1. Open http://localhost:3000 in your browser"
echo "2. Upload a CSV file (drag & drop or click)"
echo "3. Train models and download reports"
echo ""
echo "${YELLOW}🛑 To Stop:${NC}"
echo "   Run: ${BLUE}./stop.sh${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Logs:"
echo "  Backend:  ${BLUE}tail -f backend.log${NC}"
echo "  Frontend: ${BLUE}tail -f frontend.log${NC}"
echo ""

# Save PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

echo "${GREEN}✨ Services are running in the background.${NC}"
echo "${GREEN}✨ Use ./stop.sh to stop all services.${NC}"
echo ""
