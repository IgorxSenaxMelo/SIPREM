@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente Python nao encontrado.
    echo Primeiro execute no PowerShell:
    echo python -m venv .venv
    echo .venv\Scripts\Activate.ps1
    echo pip install -r requirements.txt
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
streamlit run app.py
pause
