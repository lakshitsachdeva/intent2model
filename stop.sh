#!/bin/bash

# Stop Intent2Model services

echo "🛑 Stopping Intent2Model services..."

# Kill by PIDs if they exist
if [ -f .backend.pid ]; then
    kill $(cat .backend.pid) 2>/dev/null || true
    rm .backend.pid
fi

if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null || true
    rm .frontend.pid
fi

# Kill by process name
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true

echo "✅ All services stopped"
