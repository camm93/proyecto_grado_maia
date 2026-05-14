@echo off
REM start.bat — Arranque rapido para Windows
REM Uso: start.bat [puerto]

set PORT=%1
if "%PORT%"=="" set PORT=8000

echo ===============================================
echo   MAIA Proyecto rethoric Despliegue Rapido
echo ===============================================

cd /d "%~dp0"

if not exist venv (
    echo Creando entorno virtual...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Instalando/verificando dependencias...
pip install -r backend\requirements.txt 

echo Iniciando servidor en http://localhost:%PORT%
echo Docs API: http://localhost:%PORT%/docs
echo Pulsa Ctrl+C para detener
echo ===============================================

uvicorn backend.main:app --host 0.0.0.0 --port %PORT% --reload
