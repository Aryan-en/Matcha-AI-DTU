@echo off
echo ========================================
echo  Matcha AI Inference Server Setup (GPU)
echo ========================================
echo.

cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install GPU dependencies
echo Installing GPU dependencies (PyTorch CUDA 12.1)...
pip install --upgrade pip
pip install -r requirements-gpu.txt

:: Set environment variables
set ORCHESTRATOR_URL=http://localhost:4000
set GEMINI_API_KEY=AIzaSyBBhthb4OoLTZQS6aWKv0QU5ix-13ew6V8

:: Check CUDA
echo.
echo Checking GPU availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo ========================================
echo  Starting Inference Server on port 8000
echo ========================================
echo.

:: Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
