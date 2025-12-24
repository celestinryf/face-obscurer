@echo off
echo Starting Privacy Engine API...
echo.

REM Check if venv exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: No virtual environment found.
    echo Run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    echo.
)

echo Starting server at http://localhost:8000
echo API docs at http://localhost:8000/docs
echo Press Ctrl+C to stop.
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000