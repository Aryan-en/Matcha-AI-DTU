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
if exist requirements-gpu.txt (
    pip install -r requirements-gpu.txt
) else (
    echo WARNING: requirements-gpu.txt not found, using requirements.txt...
    if not exist requirements.txt (
        echo ERROR: requirements.txt not found
        pause
        exit /b 1
    )
    pip install -r requirements.txt
)
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)

:: Set environment variables
set ORCHESTRATOR_URL=http://localhost:4000/api/v1

:: Load GEMINI_API_KEY from environment or .env file
if defined GEMINI_API_KEY (
    echo GEMINI_API_KEY loaded from environment
) else if exist ".env" (
    echo Loading .env file...
    for /f "tokens=*" %%i in ('findstr /i "GEMINI_API_KEY" .env') do set "%%i"
)

if not defined GEMINI_API_KEY (
    echo WARNING: GEMINI_API_KEY not set. Gemini features will be unavailable.
    echo To enable: set GEMINI_API_KEY=your_api_key or add to .env file
)

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
