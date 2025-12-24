@echo off
echo ========================================
echo Privacy Engine - Installation Script
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [1/5] Python found:
python --version
echo.

REM Create venv
echo [2/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo Virtual environment created.
echo.

REM Activate venv
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install cmake first
echo [4/5] Installing cmake (required for dlib)...
pip install cmake
echo.

REM Install requirements
echo [5/5] Installing dependencies (this may take 5-10 minutes for dlib)...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Installation failed!
    echo ========================================
    echo.
    echo If dlib failed to compile, you need Visual Studio Build Tools:
    echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    echo Select "Desktop development with C++" during installation.
    echo Then restart this script.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo To run the server:
echo   1. Open PowerShell in this folder
echo   2. Run: .\venv\Scripts\Activate
echo   3. Run: uvicorn app.main:app --reload --port 8000
echo.
echo Or just double-click run_server.bat
echo.
echo API will be at: http://localhost:8000
echo API docs at: http://localhost:8000/docs
echo.
pause