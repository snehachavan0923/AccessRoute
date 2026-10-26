# AccessRoute

**AI-Based Accessibility and Traffic-Aware Route Recommendation for Public Spaces Using Computer Vision**

AccessRoute is a web application that compares pedestrian routes using OpenStreetMap walking networks and simulated traffic data for Electronic City, Bengaluru.

## Prerequisites

Install the following tools before setting up the project:

- Git
- Python 3.12
- Node.js and npm
- VS Code (recommended)

### Install prerequisites on macOS
-------------------
If Homebrew is already installed, run:

```bash
brew install git python@3.12 node
```

If you do not have Homebrew, follow the instructions at https://brew.sh/.

Verify the installations:

```bash
git --version
python3.12 --version
node -v
npm -v
```

### Install prerequisites on Windows

**1. Install Git**

Download and install Git from:
https://git-scm.com/download/win

**2. Install Python 3.12**

Download Python from:
https://www.python.org/downloads/

During installation, select **Add Python to PATH**.

Verify the installation in Command Prompt or PowerShell:

```powershell
py -3.12 --version
```

**3. Install Node.js and npm**

Download the LTS version from:
https://nodejs.org/

Verify the installation:

```powershell
node -v
npm -v
```

**4. Install VS Code (recommended)**

Download it from:
https://code.visualstudio.com/

## 1. Clone the Repository

Open Terminal on macOS or PowerShell on Windows.

Run:

```bash
git clone https://github.com/snehachavan0923/AccessRoute.git
cd AccessRoute
```

To open the project in VS Code, run:

```bash
code .
```

## 2. Set Up the Backend

### macOS

Run these commands from the project root (`AccessRoute/`):

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Start the backend:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Windows

Open PowerShell in the project root (`AccessRoute/`).

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Start the backend:

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Keep the backend terminal open.

**Backend URLs:**

- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## 3. Set Up the Frontend

Open a **second terminal** in the project root.

### macOS and Windows

Run:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed in the terminal, usually:

http://localhost:5173

Keep both the backend and frontend servers running while using the application.

## 4. Configure the Traffic Dataset

AccessRoute uses simulated traffic data for Electronic City, Bengaluru. The dataset is not included in this GitHub repository.

1. Ask the project team for the dataset download link.
2. Download and extract the dataset.
3. Configure the dataset location if required by the backend.

The documented default dataset location on macOS is:

```text
~/Downloads/electronic-city-traffic-clean
```

On Windows, use the actual folder path where the dataset is extracted and configure it according to the backend's environment-variable settings.

If the dataset is unavailable or its path is not configured correctly, traffic-aware features may be limited.

## 5. Environment Configuration

The project may use environment variables for API URLs and routing/cache settings.

Check the existing `.env` file and backend configuration before changing any values.

The frontend typically connects to the backend at:

```text
http://localhost:8000
```

**Important:** Do not commit personal `.env` files, local file paths, or secrets to GitHub.

## 6. Troubleshooting

**Python or Uvicorn command not found**

Make sure Python 3.12 is installed, the virtual environment is activated, and the terminal is in the project root.

**PowerShell blocks virtual-environment activation**

Use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

**Frontend cannot connect to the backend**

Confirm that the backend is running on port `8000`. Check the browser console for API or CORS errors.

**Port 8000 is already in use**

Stop the other process using the port before restarting the backend.

**Map tiles are not loading**

Check your internet connection. OpenStreetMap tiles require network access.

## Current Scope

The application currently focuses on pedestrian route planning, route comparison, and simulated traffic information. Live camera-based computer-vision detection is not required for the basic setup.