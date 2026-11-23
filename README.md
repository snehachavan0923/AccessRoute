# AccessRoute

**AI-Based Accessibility and Traffic-Aware Route Recommendation for Public Spaces Using Computer Vision**

AccessRoute is a full-stack system designed to provide safe, traffic-aware, and accessibility-conscious pedestrian routing in public spaces. It pairs a pedestrian routing engine using OpenStreetMap network graphs with a real-time Computer Vision analysis pipeline powered by YOLOv11n and ByteTrack.

---

## Presentation Overview & Demonstration Modules

For presentation and evaluation, AccessRoute has two demonstrable, fully functional subsystems:

1. **Pedestrian Route Recommendation & Comparison Engine**
   - **Shortest Walking Route Baseline:** Computes the true physical shortest distance path using Dijkstra's algorithm on OpenStreetMap pedestrian networks.
   - **Traffic-Aware Alternative Route:** Evaluates route impedance using the **simulated Electronic City, Bengaluru traffic dataset** fallback when no live calibrated camera observations exist.
   - **Step-Free / Reduced Mobility Mode:** Evaluates pathway constraints (stairs, barriers). *Disclosure: Accessibility coverage is currently incomplete; unmonitored pathways have unknown status and are never assumed wheelchair-accessible.*
   - **Route Explainability:** Clear side-by-side cost breakdowns, distance comparison, and explicit data source attribution (`Simulated Dataset` vs. `Vision Traffic`).

2. **Computer Vision & Motion Diagnostics Subsystem**
   - **Area A (Video Analysis):** Real YOLOv11n neural-network vehicle detection and ByteTrack multi-object tracking on traffic video footage.
   - **Area B (Motion Diagnostics):** Measures per-vehicle trajectory lifespans, cumulative and net pixel displacement ($\text{px}$), average movement rate ($\text{px/s}$), stationary classification heuristics, and duration-weighted scene movement.
   - **Area C (Traffic & Route Impact):** Displays the validation safeguard gates and confirms that arbitrary uploaded or sample video **does not automatically alter route recommendations** because camera calibration (homography), directed road mapping, and verified congestion estimation are safety requirements.

---

## Running AccessRoute with GitHub Codespaces

AccessRoute is fully configured for [GitHub Codespaces](https://github.com/features/codespaces). Teammates can launch the entire stack in cloud-hosted Linux containers without installing Python, Node.js, system libraries, or virtual environments locally.

### 1. Launch a Codespace
1. Navigate to your GitHub repository in your browser.
2. Click the green **Code** button, select the **Codespaces** tab, and click **Create codespace on main** (or click **+**).
3. GitHub Codespaces will build the container based on `.devcontainer/devcontainer.json`, which automatically runs `.devcontainer/setup.sh` to:
   - Configure Python 3.12 and Node.js 22.
   - Install required Linux multimedia libraries (`ffmpeg`, `libsm6`, `libxext6`).
   - Create a local virtual environment (`.venv`) and install all backend and vision dependencies.
   - Install frontend npm packages (`frontend/node_modules`).
   - Pre-download required neural network weights (`yolo11n.pt`).
   - Initialize the local SQLite database schema.

### 2. Configure TomTom API Key (Optional Secret)
If you want live TomTom traffic flow integration instead of fallback traffic data:
1. In your GitHub repository, go to **Settings** > **Secrets and variables** > **Codespaces**.
2. Click **New repository secret**.
3. Name: `TOMTOM_API_KEY`
4. Value: Your TomTom API key.
5. Launch or restart your Codespace. The key will be automatically injected into the container environment. If omitted, AccessRoute gracefully falls back to the built-in simulated dataset without crashing.

### 3. Start the Application
You can start the full stack in any of the following ways:

- **Option A — One-Click Command (Terminal):**
  ```bash
  ./start.sh
  ```
  *(Press `Ctrl+C` to cleanly stop both backend and frontend servers).*

- **Option B — VS Code Task:**
  Press `Ctrl+Shift+P` (or `Cmd+Shift+P`), select **Tasks: Run Task**, and choose **Start AccessRoute (All)**.

- **Option C — Manual Terminals:**
  - Backend: `source .venv/bin/activate && uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`
  - Frontend: `npm --prefix frontend run dev -- --host 0.0.0.0`

### 4. Access the Application
- Codespaces will automatically forward:
  - **Port 5173** (Vite Frontend): Click the notification or open the **Ports** panel to access the web app.
  - **Port 8000** (FastAPI Backend & Interactive Swagger Docs at `/docs`).
- The frontend is preconfigured to route API calls seamlessly through Vite's local proxy to avoid CORS or port conflicts in Codespace browser previews.

---

## System Requirements & Prerequisites (macOS)

- **Operating System:** macOS Ventura (13+) or newer (Apple Silicon M1/M2/M3/M4 or Intel).
- **Python:** Python 3.12 (`python3.12`)
- **Node.js:** Node.js 20+ and `npm`
- **Git**
- **Web Browser:** Modern Chromium or Safari browser with WebGL/Canvas support for Leaflet maps.

### Install Prerequisites on macOS via Homebrew

If Homebrew is installed, run:
```bash
brew install git python@3.12 node
```

Verify your installed versions:
```bash
python3.12 --version   # Should report Python 3.12.x
node -v                # Should report v20.x or newer
npm -v                 # Should report v10.x or newer
git --version
```

---

## Project Structure

```text
AccessRoute/
├── backend/
│   ├── main.py                     # FastAPI application entrypoint and CORS setup
│   ├── routing/                    # Dijkstra shortest path, traffic cost formulas, router API
│   ├── traffic/                    # Electronic City dataset loader and VisionTrafficAdapter
│   ├── accessibility/              # Accessibility data contracts and obstacle adapter
│   ├── vision/                     # YOLO11n service, ByteTrack tracking, motion analyzer
│   │   └── weights/
│   │       └── yolo11n.pt          # Pre-trained YOLO11n model weights (5.6 MB, included)
│   ├── requirements.txt            # Core backend routing dependencies
│   ├── requirements-vision.txt     # Vision & PyTorch dependencies
│   └── tests/                      # Automated test suite (269 unit & integration tests)
├── frontend/
│   ├── src/                        # React components, Leaflet map, and API services
│   ├── package.json                # Frontend dependencies (React 19, Vite, Tailwind CSS v4)
│   ├── vite.config.js              # Vite bundler configuration
│   └── index.html                  # Frontend HTML entrypoint
├── data/
│   └── vision/
│       └── WhatsApp Video 2026-10-26 at 9.29.19 AM.mp4  # Sample traffic footage (920 KB, included)
├── research/                       # Research notebooks and experiment reports
├── .env.example                    # Template environment variables
└── README.md                       # Setup and presentation documentation
```

---

## Step-by-Step macOS Setup Instructions

### 1. Extract and Enter the Project Root

```bash
unzip AccessRoute.zip -d AccessRoute
cd AccessRoute
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(No API keys or passwords are required for local operation. Sensible defaults are pre-configured.)*

---

### 3. Set Up the Python Backend

From the project root (`AccessRoute/`):

1. **Create a Python 3.12 virtual environment:**
   ```bash
   python3.12 -m venv .venv
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Upgrade pip and install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r backend/requirements.txt
   pip install -r backend/requirements-vision.txt
   ```

4. **Verify the installation by running backend tests:**
   ```bash
   python -m unittest discover -s backend/tests
   ```
   *Expected output:* `Ran 269 tests ... OK`

5. **Start the FastAPI backend server:**
   ```bash
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```
   *Keep this terminal running.*

   - **Backend API Base:** http://127.0.0.1:8000
   - **Interactive API Documentation:** http://127.0.0.1:8000/docs
   - **Health Check Endpoint:** http://127.0.0.1:8000/health

---

### 4. Set Up the React Frontend

Open a **second terminal window** and navigate to the project root (`AccessRoute/`):

1. **Enter the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node.js packages:**
   ```bash
   npm install
   ```

3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```

4. **Open the web application in your browser:**
   ```text
   http://localhost:5173
   ```
   *Keep both the backend and frontend terminals running simultaneously during your demo.*

---

## External Files & Datasets

### 1. YOLO11n Model Weights (`yolo11n.pt`)
- **Status:** **Included** in the distributable ZIP inside `backend/vision/weights/yolo11n.pt` (~5.6 MB).
- **Fallback / Re-download:** If deleted or missing, the backend will automatically re-download standard `yolo11n.pt` weights on first analysis run if connected to the internet.

### 2. Sample Traffic Video
- **Status:** **Included** in the distributable ZIP inside `data/vision/` (~920 KB).
- Pre-loaded automatically as the default sample video in Area A.

### 3. Electronic City Traffic Dataset (`electronic-city-traffic-clean`)
- **Status:** **External academic dataset** (~125 MB of binary NumPy matrices).
- **Default macOS Location:**
  ```text
  ~/Downloads/electronic-city-traffic-clean
  ```
- **How to obtain:** Request the dataset folder archive from the project team. Extract it to your macOS `~/Downloads/` folder.
- **Custom Location:** If stored in another folder, specify the path in your `.env` file:
  ```env
  TRAFFIC_DATASET_DIR=/path/to/electronic-city-traffic-clean
  ```
- **Expected Directory Structure:**
  ```text
  electronic-city-traffic-clean/
  ├── graph/
  │   ├── edges_final.csv
  │   └── nodes_final.csv
  └── traffic/
      ├── congestion_matrix.npy
      ├── speed_matrix.npy
      ├── timestamps.csv
      └── vehicle_matrix.npy
  ```
- **Graceful Fallback:** If the dataset folder is not present on disk, the system displays a clear notice in the route comparison panel and safely falls back to the shortest physical pedestrian path. All Computer Vision and motion diagnostic capabilities remain 100% functional.

---

## Presentation Walkthrough & Demonstration Flow

1. **Demonstrate Route Planning & Traffic Comparison:**
   - On the main Route Planner page, select an origin and destination within Electronic City (e.g. *Velankani Tech Park* to *Wipro Gate 5*).
   - Show the **Shortest Walking Route Baseline** (pure physical distance) and the **Recommended Route** (incorporating simulated dataset traffic avoidance).
   - Switch between **Morning Peak (08:30 AM)** and **Night Free-Flow (02:00 AM)** to show how simulated dataset congestion affects route recommendations.
   - Toggle **Step-Free / Reduced** mobility mode to demonstrate barrier-avoidance routing, and highlight the disclaimer explaining incomplete accessibility coverage.

2. **Demonstrate Computer Vision Pipeline:**
   - Click **Computer Vision Pipeline** in the top navigation bar.
   - **Area A (Video Analysis):** Select the included sample video and click **Run Computer Vision Analysis**. Observe live YOLO11n detection bounding boxes, confidence scores, ByteTrack tracking IDs, and class breakdown (cars, motorcycles, buses, trucks).
   - **Area B (Motion Diagnostics):** Inspect individual vehicle trajectories, displacement in pixels ($\text{px}$), average movement rate in pixels per second ($\text{px/s}$), stationary classification status, and duration-weighted scene movement.
   - **Area C (Traffic & Route Impact):** Review the safeguard validation checklist. Explain why uncalibrated video does not alter route recommendations, and confirm that the router safely uses the simulated dataset fallback.

---

## Troubleshooting (macOS)

- **Port 8000 or 5173 already in use:**
  ```bash
  lsof -ti :8000 | xargs kill -9
  lsof -ti :5173 | xargs kill -9
  ```
- **Virtual environment activation failure (`zsh: permission denied`):**
  Ensure you run `source .venv/bin/activate` (with `source`, not `./.venv/bin/activate`).
- **OpenCV headless errors:**
  AccessRoute uses `opencv-python-headless` and `ultralytics-opencv-headless` to eliminate macOS GUI framework conflicts. Ensure you install dependencies via:
  ```bash
  pip install -r backend/requirements-vision.txt
  ```
- **Map tiles not rendering:**
  Verify your Mac has an active internet connection so Leaflet can fetch OpenStreetMap tiles.

---

## Verification Commands for Teammates

Teammates can verify their local setup with these commands:

```bash
# 1. Run all backend tests (269 tests)
python -m unittest discover -s backend/tests

# 2. Run frontend linter
cd frontend && npm run lint

# 3. Run frontend production build
cd frontend && npm run build
```