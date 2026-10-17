@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Primeira execucao: criando ambiente Python...
    python -m venv .venv
    if errorlevel 1 (
        echo Python nao encontrado. Instale o Python 3.11 ou 3.12 marcando "Add Python to PATH".
        pause
        exit /b 1
    )
    call ".venv\Scripts\activate.bat"
    pip install -r requirements.txt
) else (
    call ".venv\Scripts\activate.bat"
)
streamlit run app.py
pause
