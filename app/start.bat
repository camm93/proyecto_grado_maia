@echo off
REM start.bat - Arranque del backend MAIA (Windows).
REM ============================================================================
REM ASCII-only: este archivo no usa caracteres no-ASCII para que se vea
REM correctamente en CMD, PowerShell y Windows Terminal sin importar la
REM codepage del sistema. El chcp 65001 de abajo intenta UTF-8 igualmente
REM por si la consola del usuario lo soporta, pero el contenido es puro
REM ASCII como fallback.
REM ============================================================================
chcp 65001 >nul 2>nul

setlocal EnableDelayedExpansion

set PORT=8000
set DO_INSTALL=true
set DO_SYNC=true
set DO_RELOAD=true
set VERIFY_ONLY=false

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--port"         ( set PORT=%~2 & shift & shift & goto parse_args )
if /i "%~1"=="--no-reload"    ( set DO_RELOAD=false & shift & goto parse_args )
if /i "%~1"=="--skip-install" ( set DO_INSTALL=false & shift & goto parse_args )
if /i "%~1"=="--skip-sync"    ( set DO_SYNC=false & shift & goto parse_args )
if /i "%~1"=="--verify-only"  ( set VERIFY_ONLY=true & shift & goto parse_args )
if /i "%~1"=="--help"         ( goto show_help )
if /i "%~1"=="-h"             ( goto show_help )
echo Argumento desconocido: %~1 -- ver --help
exit /b 1

:show_help
echo start.bat - Arranque del backend MAIA (Windows).
echo.
echo Hace, en este orden:
echo   (a) pip install -r backend\requirements.txt   (idempotente)
echo   (b) Sincronizacion + verificacion de pesos via model_sync
echo       - 1er arranque: descarga 3 ZIPs desde Drive (~3.9 GB, ~5-8 min)
echo       - Re-arranques: solo verifica SHAs (~5-10 s)
echo   (c) uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo.
echo Uso:
echo   start.bat                       Arranque normal
echo   start.bat --port 8080           Cambia el puerto (default 8000)
echo   start.bat --no-reload           Sin --reload (recomendado en produccion)
echo   start.bat --skip-install        Omite el pip install
echo   start.bat --skip-sync           Omite model_sync (asume pesos OK)
echo   start.bat --verify-only         Solo verifica integridad y sale
echo   start.bat --help                Esta ayuda
echo.
echo Variables de entorno relevantes (ver backend\utils\model_sync.py):
echo   MODELS_DIR              default .\models
echo   MANIFEST_PATH           default .\release_manifest.json
echo   DOWNLOAD_FAIL_POLICY    abort ^| warn (default abort)
echo   T1_PRELOAD              true ^| false (default true)
echo   T2_PRELOAD              true ^| false (default true)
exit /b 0

:args_done

cd /d "%~dp0"

echo ============================================================
echo   MAIA -- Analisis Retorico (v7.8.2)
echo ============================================================

REM -- (a) Dependencias --------------------------------------------------------
if "%DO_INSTALL%"=="true" (
    echo.
    echo [a/c] Instalando/verificando dependencias...
    if not exist venv (
        echo   Creando entorno virtual .\venv ...
        python -m venv venv
    )
    call venv\Scripts\activate.bat
    REM Upgrade pip via 'python -m pip' (NO 'pip install --upgrade pip'
    REM directo: pip se rehusa a auto-modificarse cuando lo invoca el .exe
    REM en Windows). El '2^>nul ^|^| ver ^>nul' neutraliza el ERRORLEVEL si
    REM falla -- un pip viejo no es bloqueante para el resto del install.
    python -m pip install -q --upgrade pip 2>nul || ver >nul
    pip install -q -r backend\requirements.txt
    if errorlevel 1 (
        echo   FAIL -- pip install fallo. Abortando.
        exit /b 1
    )
    echo   OK -- dependencias listas
) else (
    if exist venv ( call venv\Scripts\activate.bat )
    echo [a/c] pip install omitido (--skip-install^)
)

REM -- (b) Sincronizacion de modelos -------------------------------------------
if "%MODELS_DIR%"==""    set MODELS_DIR=%CD%\models
if "%MANIFEST_PATH%"=="" set MANIFEST_PATH=%CD%\release_manifest.json

if "%VERIFY_ONLY%"=="true" (
    echo.
    echo [verify-only] Verificando integridad SHA256 de pesos en %MODELS_DIR%
    python -m backend.utils.model_sync verify
    exit /b %ERRORLEVEL%
)

if "%DO_SYNC%"=="true" (
    echo.
    echo [b/c] Sincronizando pesos de modelos...
    echo   MODELS_DIR=%MODELS_DIR%
    echo   MANIFEST_PATH=%MANIFEST_PATH%
    echo   (1er arranque: ~5-8 min descargando ~3.9 GB desde Drive^)
    echo   (re-arranques: ~5-10 s, solo verifica SHAs^)
    python -m backend.utils.model_sync ensure_all
    if errorlevel 1 (
        echo FAIL -- model_sync fallo. Abortando.
        exit /b 1
    )
) else (
    echo [b/c] model_sync omitido (--skip-sync^). Asumiendo pesos ya OK.
)

REM -- (c) uvicorn -------------------------------------------------------------
echo.
echo [c/c] Arrancando uvicorn en http://0.0.0.0:%PORT%
echo   Docs:   http://localhost:%PORT%/docs
echo   Health: http://localhost:%PORT%/health
echo   Ctrl+C para detener.
echo ============================================================

if "%DO_RELOAD%"=="true" (
    uvicorn backend.main:app --host 0.0.0.0 --port %PORT% --reload
) else (
    uvicorn backend.main:app --host 0.0.0.0 --port %PORT%
)
