@echo off
setlocal enabledelayedexpansion

echo =========================================
echo   KRIXION Hate Speech Detection - Setup
echo =========================================

set ROOT_DIR=%~dp0
set VENV_DIR=%ROOT_DIR%.venv
set VENDOR_DIR=%ROOT_DIR%vendor
set MODELS_TRANSFORMER_DIR=%ROOT_DIR%models\transformer\distilbert_local
set SCRIPTS_DIR=%ROOT_DIR%scripts

REM 1) Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install Python 3.10+ and re-run.
    exit /b 2
)

REM 2) Create virtual environment if missing
if not exist "%VENV_DIR%" (
    echo Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
) else (
    echo Virtual environment already exists at %VENV_DIR%
)

REM 3) Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM 4) Install dependencies (offline-first)
if exist "%VENDOR_DIR%\*.whl" (
    echo Installing dependencies from vendor/ (offline mode)...
    python -m pip install --no-index --find-links "%VENDOR_DIR%" -r requirements.txt
) else (
    echo vendor/ not found or empty -- installing from PyPI (requires internet)...
    python -m pip install -r requirements.txt
)

REM 5) Check & download transformer model
if not exist "%MODELS_TRANSFORMER_DIR%" (
    if exist "%SCRIPTS_DIR%\download_transformer.py" (
        echo Transformer model not found locally -- attempting to download (this may require internet)...
        python "%SCRIPTS_DIR%\download_transformer.py"
    ) else (
        echo Transformer model missing and download script not found.
        echo Place a transformer model under models\transformer\distilbert_local or add scripts\download_transformer.py
    )
) else (
    echo Transformer model present at %MODELS_TRANSFORMER_DIR%
)

REM 6) Initialize local SQLite DB
echo Initializing local SQLite DB (data/app.db)...
python -c "from src.utils.db import init_db; init_db(); print('DB initialized -> data/app.db')"

REM 7) Run preflight check if present
if exist "%SCRIPTS_DIR%\preflight_check.py" (
    echo Running preflight check...
    python "%SCRIPTS_DIR%\preflight_check.py"
) else (
    echo Preflight check script not found at scripts\preflight_check.py -- skipping.
)

echo.
echo =========================================
echo   INSTALLATION COMPLETE
echo.
echo To run the app now (Windows):
echo   .venv\Scripts\activate.bat
echo   python app.py
echo.
echo Notes:
echo  - For fully offline operation ensure vendor/ contains wheel files and
echo    models/transformer/distilbert_local exists (or the download script can fetch it).
echo  - If transformer download fails due to Hugging Face gating, obtain the model manually
echo    and place it at models/transformer/distilbert_local/
echo =========================================

pause
exit /b 0
