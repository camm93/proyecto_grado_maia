#!/usr/bin/env bash
# update_duckdns.sh - Auto-actualiza DuckDNS con la IP pública actual.
#
# Útil si NO usás Elastic IP y querés que al reiniciar la EC2 el dominio
# se ajuste solo a la nueva IP.
#
# Setup (una sola vez):
#   1. Editar las variables DUCKDNS_DOMAIN y DUCKDNS_TOKEN abajo
#      (token disponible en https://www.duckdns.org/ después de loguear)
#   2. Copiar este script a /usr/local/bin/update_duckdns.sh
#   3. Marcarlo ejecutable: sudo chmod +x /usr/local/bin/update_duckdns.sh
#   4. Habilitarlo al boot vía systemd timer (ver instrucciones abajo)
# ============================================================================
set -u

# ────────── CONFIGURAR ESTO ───────────────────────────────────────────────
DUCKDNS_DOMAIN="maia-g2"        # solo el sub, sin ".duckdns.org"
DUCKDNS_TOKEN="REEMPLAZAR-CON-TU-TOKEN"
# ──────────────────────────────────────────────────────────────────────────

LOG_FILE="/var/log/update_duckdns.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG_FILE"
}

# Obtener IP pública actual
CURRENT_IP=$(curl -s --max-time 10 checkip.amazonaws.com)
if [[ -z "$CURRENT_IP" ]]; then
    log "ERROR: no se pudo obtener IP publica"
    exit 1
fi

log "IP publica actual: $CURRENT_IP"

# Llamar a la API de DuckDNS (no expone token en logs si fallás)
RESPONSE=$(curl -s --max-time 10 \
    "https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=${CURRENT_IP}")

if [[ "$RESPONSE" == "OK" ]]; then
    log "DuckDNS actualizado: ${DUCKDNS_DOMAIN}.duckdns.org -> $CURRENT_IP"
    exit 0
else
    log "ERROR: DuckDNS respondio: $RESPONSE"
    exit 1
fi

# ────────── Instrucciones de instalación con systemd timer ─────────────────
#
# Crear /etc/systemd/system/update-duckdns.service:
# ----
# [Unit]
# Description=Update DuckDNS with current public IP
# After=network-online.target
# Wants=network-online.target
#
# [Service]
# Type=oneshot
# ExecStart=/usr/local/bin/update_duckdns.sh
# ----
#
# Crear /etc/systemd/system/update-duckdns.timer:
# ----
# [Unit]
# Description=Update DuckDNS periodically and at boot
#
# [Timer]
# OnBootSec=30s
# OnUnitActiveSec=10min
# Persistent=true
#
# [Install]
# WantedBy=timers.target
# ----
#
# Activar:
#   sudo systemctl daemon-reload
#   sudo systemctl enable --now update-duckdns.timer
#
# Verificar:
#   sudo systemctl list-timers update-duckdns.timer
#   sudo journalctl -u update-duckdns -n 20
#
