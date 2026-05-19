#!/usr/bin/env bash
# install_systemd.sh - Instala MAIA como service de systemd en la EC2.
#
# Hace que el backend arranque automáticamente al boot y sobreviva
# cierre de sesión SSH. Tras instalarlo no necesitás más tmux.
#
# Uso:
#   sudo ./scripts/install_systemd.sh
#
# Comandos post-instalación útiles:
#   sudo systemctl status  maia    Ver estado actual
#   sudo systemctl start   maia    Arrancar
#   sudo systemctl stop    maia    Detener
#   sudo systemctl restart maia    Reiniciar
#   sudo systemctl disable maia    No arrancar al boot
#   sudo journalctl -u maia -f     Ver logs en tiempo real
#   sudo journalctl -u maia --since "1 hour ago"
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_FILE="$SCRIPT_DIR/maia.service"
TARGET="/etc/systemd/system/maia.service"

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: Este script requiere sudo. Re-correr con:"
    echo "  sudo $0"
    exit 1
fi

if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "ERROR: No se encuentra $SERVICE_FILE"
    exit 1
fi

# Detectar si el proyecto NO está en /home/ec2-user/app y avisar
EXPECTED_DIR="/home/ec2-user/app"
if [[ "$PROJECT_DIR" != "$EXPECTED_DIR" ]]; then
    echo "ADVERTENCIA: El proyecto esta en $PROJECT_DIR pero maia.service"
    echo "             apunta a $EXPECTED_DIR. Voy a parchear el .service."
    # Crear copia parcheada con sed
    sed "s|$EXPECTED_DIR|$PROJECT_DIR|g" "$SERVICE_FILE" > /tmp/maia.service.patched
    SERVICE_FILE=/tmp/maia.service.patched

    # Detectar usuario actual (ec2-user, ubuntu, etc.) y reemplazar
    REAL_USER="${SUDO_USER:-ec2-user}"
    sed -i "s|User=ec2-user|User=$REAL_USER|g; s|Group=ec2-user|Group=$REAL_USER|g" "$SERVICE_FILE"
    echo "             User/Group = $REAL_USER"
fi

# Verificar que start.sh es ejecutable
if [[ ! -x "$PROJECT_DIR/start.sh" ]]; then
    echo "Marcando $PROJECT_DIR/start.sh como ejecutable..."
    chmod +x "$PROJECT_DIR/start.sh"
fi

# Verificar que existe .env
if [[ ! -f "$PROJECT_DIR/.env" ]]; then
    echo "ERROR: No se encuentra $PROJECT_DIR/.env"
    echo "       Crear el archivo .env con HF_TOKEN, OPENAI_API_KEY, etc. antes."
    exit 1
fi

echo "Instalando service file en $TARGET..."
cp "$SERVICE_FILE" "$TARGET"
chmod 644 "$TARGET"

echo "Recargando systemd..."
systemctl daemon-reload

echo "Habilitando arranque automatico al boot..."
systemctl enable maia.service

echo
echo "============================================================"
echo "  MAIA service instalado."
echo "============================================================"
echo
echo "Iniciar AHORA:"
echo "    sudo systemctl start maia"
echo
echo "Ver logs en tiempo real:"
echo "    sudo journalctl -u maia -f"
echo
echo "Estado:"
echo "    sudo systemctl status maia"
echo
echo "Si ya tenias uvicorn corriendo manualmente (./start.sh o tmux),"
echo "matalo primero antes de hacer systemctl start, o se va a chocar"
echo "el puerto 8000:"
echo "    ps aux | grep uvicorn"
echo "    kill <PID>"
echo
