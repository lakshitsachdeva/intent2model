#!/bin/bash

# Start backend
echo "starting backend..."
cd backend
export GEMINI_API_KEY=AIzaSyDc6lDoHJmM1_YEP4XPdl17349eKvg0JAE
python main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "starting frontend..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "backend running on http://localhost:8000"
echo "frontend running on http://localhost:3000"
echo ""
echo "press ctrl+c to stop both servers"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
