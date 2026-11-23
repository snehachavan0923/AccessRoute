#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

echo "=================================================="
echo "  Setting up AccessRoute in GitHub Codespaces...  "
echo "=================================================="

# 1. System packages for OpenCV and video processing
if command -v apt-get >/dev/null 2>&1; then
    echo "[1/7] Installing system dependencies (ffmpeg, libsm6, libxext6)..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends ffmpeg libsm6 libxext6
else
    echo "[1/7] apt-get not available, skipping system package installation."
fi

# 2. Environment file
echo "[2/7] Checking environment configuration..."
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
else
    echo ".env already exists."
fi

# 3. Python virtual environment
echo "[3/7] Setting up Python virtual environment..."
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# 4. Backend dependencies
echo "[4/7] Installing backend dependencies..."
pip install -r backend/requirements.txt -q
pip install -r backend/requirements-vision.txt -q
pip install pytest -q

# 5. Frontend dependencies
echo "[5/7] Installing frontend dependencies..."
if [ -f frontend/package-lock.json ]; then
    npm --prefix frontend ci --prefer-offline --no-audit || npm --prefix frontend install
else
    npm --prefix frontend install
fi

# 6. Directories and model weights
echo "[6/7] Ensuring data directories and weights..."
mkdir -p cache/graphs
mkdir -p backend/vision/weights

WEIGHTS_PATH="backend/vision/weights/yolo11n.pt"
if [ ! -f "${WEIGHTS_PATH}" ]; then
    echo "Downloading YOLO11n weights into ${WEIGHTS_PATH}..."
    curl -fsSL "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" -o "${WEIGHTS_PATH}" || {
        echo "Warning: Direct download failed, attempting via Python..."
        PYTHONPATH=. python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt', '${WEIGHTS_PATH}')" || true
    }
fi

# 7. Database initialization
echo "[7/7] Initializing SQLite database..."
PYTHONPATH=. python -c "from backend.database.initialize import initialize_database; initialize_database()"

echo "=================================================="
echo "  AccessRoute setup complete!                     "
echo "                                                  "
echo "  To start both backend and frontend:             "
echo "    ./start.sh                                    "
echo "  Or run the VS Code build task:                  "
echo "    Ctrl+Shift+B (or Cmd+Shift+B on macOS)         "
echo "=================================================="
