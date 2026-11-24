#!/usr/bin/env bash
set -e

# Resolve repository root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Ensure virtual environment exists and is activated
if [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
    source "${ROOT_DIR}/.venv/bin/activate"
fi

# Kill background servers on exit (Ctrl+C)
cleanup() {
    echo ""
    echo "Shutting down AccessRoute development servers..."
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "=================================================="
echo "  Starting AccessRoute (Backend + Frontend)       "
echo "=================================================="

# Check if port 8000 or 5173 is already in use by a stale process
if command -v lsof >/dev/null 2>&1; then
    PORT_8000_PID=$(lsof -Pi :8000 -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$PORT_8000_PID" ]; then
        echo "Port 8000 is occupied by stale process(es): $PORT_8000_PID. Terminating to reload fresh backend..."
        kill -9 $PORT_8000_PID 2>/dev/null || true
        sleep 1
    fi
    PORT_5173_PID=$(lsof -Pi :5173 -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$PORT_5173_PID" ]; then
        echo "Port 5173 is occupied by stale process(es): $PORT_5173_PID. Terminating to reload fresh frontend..."
        kill -9 $PORT_5173_PID 2>/dev/null || true
        sleep 1
    fi
fi

# Start FastAPI backend
echo "Starting FastAPI backend on http://0.0.0.0:8000..."
PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &

# Start Vite frontend
echo "Starting Vite frontend on http://0.0.0.0:5173..."
npm --prefix frontend run dev -- --host 0.0.0.0 &

echo ""
echo "AccessRoute servers are running!"
echo "Backend:  http://localhost:8000 (API Docs: http://localhost:8000/docs)"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both servers."
echo "=================================================="

wait
