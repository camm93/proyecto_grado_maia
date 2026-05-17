#!/bin/bash
# start.sh — Arranque rápido para Linux/macOS
# Uso: ./start.sh [puerto]

PORT=${1:-8000}
cd "$(dirname "$0")"

echo "==============================================="
echo "  MAIA Proyecto rethoric Despliegue Rapido"
echo "==============================================="

# Verificar entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Instalando/verificando dependencias..."
pip install -r backend/requirements.txt -q

echo "Iniciando servidor en http://localhost:$PORT"
echo "Docs API: http://localhost:$PORT/docs"
echo "Pulsa Ctrl+C para detener"
echo "==============================================="

uvicorn backend.main:app --host 0.0.0.0 --port $PORT --reload
