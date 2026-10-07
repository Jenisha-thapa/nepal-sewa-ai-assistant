@echo off
echo ============================================
echo   Nepal Sewa AI - Quick Start
echo ============================================
echo.

:: Check if Python exists
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

:: Set paths
set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found at %VENV_PYTHON%
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

:: Start backend in new window
echo Starting Backend (FastAPI) on port 8000...
start "Nepal Sewa AI - Backend" cmd /k "cd /d "%PROJECT_DIR%" && "%VENV_PYTHON%" server_extended.py"

:: Wait for backend to start
timeout /t 3 /nobreak >nul

:: Start frontend in new window
echo Starting Frontend (Web UI) on port 3000...
start "Nepal Sewa AI - Frontend" cmd /k "cd /d "%PROJECT_DIR%" && "%VENV_PYTHON%" serve_web.py --port 3000"

:: Wait for frontend to start
timeout /t 2 /nobreak >nul

echo.
echo ============================================
echo   Nepal Sewa AI is starting!
echo ============================================
echo.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo.
echo   Open http://localhost:3000 in your browser
echo.
echo   Press Ctrl+C in each window to stop
echo ============================================
echo.

:: Open browser
start http://localhost:3000

pause
