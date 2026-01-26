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
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Kill any existing processes
echo "🧹 Cleaning up old processes..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
sleep 2

# Set API key
export GEMINI_API_KEY=AIzaSyDc6lDoHJmM1_YEP4XPdl17349eKvg0JAE

# Start Backend
echo ""
echo "${BLUE}📦 Starting Backend...${NC}"
cd backend

# Check if Python dependencies are installed
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo "${YELLOW}⚠️  Installing Python dependencies...${NC}"
    pip3 install -r ../requirements.txt 2>&1 | tail -5
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in backend directory"
    exit 1
fi

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "${GREEN}✅ Backend is running on http://localhost:8000${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "⏳ Waiting for backend... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    else
        echo "${YELLOW}⚠️  Backend might still be starting. Check logs: tail -f backend.log${NC}"
        echo "Last 10 lines of backend.log:"
        tail -10 backend.log 2>/dev/null || echo "No log file yet"
    fi
done

# Start Frontend
echo ""
echo "${BLUE}🎨 Starting Frontend...${NC}"
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "${YELLOW}⚠️  Installing npm dependencies...${NC}"
    npm install 2>&1 | tail -10
fi

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found in frontend directory"
    exit 1
fi

npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 8

# Check if frontend is running
MAX_RETRIES=15
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "${GREEN}✅ Frontend is running on http://localhost:3000${NC}"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "⏳ Waiting for frontend... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 2
    else
        echo "${YELLOW}⚠️  Frontend might still be starting. Check logs: tail -f frontend.log${NC}"
        echo "Last 10 lines of frontend.log:"
        tail -10 frontend.log 2>/dev/null || echo "No log file yet"
    fi
done

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
echo "3. Chat with the AI:"
echo "   • Tell it which column to predict (e.g., 'variety')"
echo "   • Ask for a report: 'report' or 'show me results'"
echo "   • Make predictions: 'can you predict for me?'"
echo ""
echo "${YELLOW}💡 Example Commands:${NC}"
echo "   • 'variety' → trains model to predict variety"
echo "   • 'report' → shows charts and metrics"
echo "   • 'predict' → starts prediction flow"
echo "   • 'sepal.length: 5.1, sepal.width: 3.5' → makes prediction"
echo ""
echo "${YELLOW}🛑 To Stop:${NC}"
echo "   Press Ctrl+C or run: ./stop.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"
echo ""

# Save PIDs
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .backend.pid .frontend.pid; exit" INT TERM

echo "Press Ctrl+C to stop all services..."
wait
