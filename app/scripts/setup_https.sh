#!/usr/bin/env bash
# setup_https.sh - Configura HTTPS para MAIA en EC2 con DuckDNS + Let's Encrypt.
#
# Pasos manuales previos (UNA SOLA VEZ):
#   1. Ir a https://www.duckdns.org/ → login con Google/GitHub
#   2. Crear un subdominio gratis (ej: "maia-edwin.duckdns.org")
#   3. Apuntar el subdominio al IP público de tu EC2 (18.210.22.104)
#   4. Copiar el token que muestra DuckDNS
#
# Pasos manuales en AWS:
#   5. Abrir el Security Group de tu EC2 y permitir:
#        - TCP 80  (HTTP) desde 0.0.0.0/0  → necesario para el challenge
#        - TCP 443 (HTTPS) desde 0.0.0.0/0  → para servir tráfico final
#
# Después correr este script:
#   sudo ./scripts/setup_https.sh maia-edwin.duckdns.org tu-email@ejemplo.com
#
# El cert se renueva solo cada 60 días vía systemd timer (certbot-renew.timer).
# ============================================================================
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: Este script requiere sudo."
    exit 1
fi

if [[ $# -lt 2 ]]; then
    echo "Uso: sudo $0 <dominio> <email>"
    echo "Ej:  sudo $0 maia-edwin.duckdns.org edwin@ejemplo.com"
    exit 1
fi

DOMAIN="$1"
EMAIL="$2"
BACKEND_PORT=8000

echo "============================================================"
echo "  Setup HTTPS para MAIA"
echo "  Dominio: $DOMAIN"
echo "  Email:   $EMAIL"
echo "  Backend: localhost:$BACKEND_PORT"
echo "============================================================"

# -- (1) Verificar que el dominio resuelve al IP publico de esta EC2 ---------
echo
echo "[1/6] Verificando DNS..."
EC2_IP=$(curl -s checkip.amazonaws.com)
DNS_IP=$(dig +short "$DOMAIN" | head -1)
echo "  IP EC2:           $EC2_IP"
echo "  IP DNS (DuckDNS): $DNS_IP"

if [[ "$EC2_IP" != "$DNS_IP" ]]; then
    echo "  ADVERTENCIA: el dominio no apunta a esta EC2."
    echo "  Verificar en https://www.duckdns.org/ que el IP sea $EC2_IP"
    echo "  Continuando de todos modos en 5s (Ctrl+C para abortar)..."
    sleep 5
else
    echo "  OK -- DNS resuelve correctamente."
fi

# -- (2) Instalar nginx (del distro) + certbot (en venv aislado) -------------
# NOTA (v7.6.2): el certbot del package manager de AL2023 (dnf install
# python3-certbot-nginx) NO funciona en sistemas con multiples Python
# coexistiendo, que es el caso de MAIA (python 3.9 del distro + python 3.11
# instalado para el backend). Las dependencias transitivas (cryptography,
# pyOpenSSL) se instalan en el site-packages "equivocado" y el binario
# /usr/bin/certbot revienta con ModuleNotFoundError al arranque.
#
# Solucion segun doc oficial certbot.eff.org para AL2023/RHEL: certbot en
# venv aislado bajo /opt/certbot, symlinkeado a /usr/bin/certbot. Trae
# desde PyPI versiones compatibles de cryptography/pyOpenSSL y no choca
# con el resto del sistema.
echo
echo "[2/6] Instalando nginx (distro) + certbot (venv aislado)..."

dnf install -y nginx >/dev/null
echo "  OK -- nginx instalado."

# Si ya hay un certbot roto del distro lo limpiamos primero
if rpm -q certbot >/dev/null 2>&1 || rpm -q python3-certbot-nginx >/dev/null 2>&1; then
    echo "  Detectado certbot del distro (legacy), removiendo..."
    dnf remove -y certbot python3-certbot python3-certbot-nginx >/dev/null 2>&1 || true
fi

# Venv aislado para certbot (no toca el Python del sistema)
if [[ ! -x /opt/certbot/bin/certbot ]]; then
    python3 -m venv /opt/certbot/
    /opt/certbot/bin/pip install --quiet --upgrade pip
    /opt/certbot/bin/pip install --quiet certbot certbot-nginx
    ln -sf /opt/certbot/bin/certbot /usr/bin/certbot
    echo "  OK -- certbot instalado en /opt/certbot (venv)."
else
    echo "  OK -- certbot ya presente en /opt/certbot."
fi

# Sanity check: certbot arranca sin tracebacks
if ! certbot --version >/dev/null 2>&1; then
    echo "  ERROR: certbot --version fallo. Revisar /opt/certbot."
    exit 1
fi
echo "  OK -- certbot $(certbot --version 2>&1 | awk '{print $2}') operativo."

# NOTA Python 3.9: certbot >=4 advierte que dropeara 3.9 en proximas
# releases. Cuando eso pase, recrear el venv con python3.11 explicito:
#   sudo systemctl stop certbot-renew.timer
#   sudo rm -rf /opt/certbot/
#   sudo python3.11 -m venv /opt/certbot/
#   sudo /opt/certbot/bin/pip install --upgrade pip certbot certbot-nginx
#   sudo ln -sf /opt/certbot/bin/certbot /usr/bin/certbot
#   sudo systemctl start certbot-renew.timer

# -- (3) Configurar nginx como reverse proxy al backend ----------------------
echo
echo "[3/6] Configurando nginx reverse proxy..."

cat > /etc/nginx/conf.d/maia.conf <<EOF
# Bloque HTTP-only inicial. Sirve el challenge ACME en
# /.well-known/acme-challenge/ y hace proxy_pass al backend en el resto.
#
# Tras correr 'certbot --nginx --redirect' (paso [4/6] de este script):
#   - certbot AGREGA un nuevo bloque "server { listen 443 ssl; ... }"
#     con ssl_certificate y ssl_certificate_key apuntando a
#     /etc/letsencrypt/live/$DOMAIN/{fullchain,privkey}.pem
#   - certbot REESCRIBE este bloque 80 para que haga return 301 a https://
#
# NOTA HISTORICA (fix v7.6.1):
# La version anterior de este script pre-creaba aqui mismo el bloque
# "server { listen 443 ssl; ... }" con las directivas ssl_certificate
# comentadas, esperando que certbot las descomentara. Eso NO funciona en
# una corrida limpia: nginx -t (validacion del paso [3/6], antes de
# emitir el cert) falla con "no ssl_certificate is defined for the
# 'listen ... ssl' directive". La solucion es no escribir nada de SSL
# aqui y delegar 100% el bloque 443 a certbot.
server {
    listen 80;
    server_name $DOMAIN;

    # Challenge ACME (Let's Encrypt) — certbot lo necesita en el paso [4/6]
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # Tamaño max del body: textos academicos pueden ser >100k chars
    client_max_body_size 10M;

    # Timeout para LLMs: GPT ~5s, Llama GPU ~3s, dejar margen para
    # textos largos con muchos parrafos
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://127.0.0.1:$BACKEND_PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSockets/SSE si en el futuro habilitamos streaming
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

mkdir -p /var/www/html
nginx -t
echo "  OK -- config nginx valida."

# Arrancar nginx (puede estar ya andando)
systemctl enable --now nginx
systemctl reload nginx
echo "  OK -- nginx corriendo."

# -- (4) Obtener cert de Let's Encrypt ---------------------------------------
echo
echo "[4/6] Obteniendo certificado de Let's Encrypt..."
echo "  Esto puede fallar si:"
echo "    - El SG no permite TCP 80 desde 0.0.0.0/0"
echo "    - El DNS no resuelve aun (puede tardar 1-2 min tras crear el sub)"
echo "    - Ya tenes 5 certs emitidos hoy para este dominio (rate limit LE)"
echo

certbot --nginx \
    -d "$DOMAIN" \
    --email "$EMAIL" \
    --agree-tos \
    --non-interactive \
    --redirect \
    || {
        echo
        echo "ERROR: certbot fallo. Revisar arriba el mensaje."
        echo "Si fue por timeout/red, esperar 2 min y reintentar:"
        echo "  sudo certbot --nginx -d $DOMAIN --email $EMAIL --agree-tos"
        exit 1
    }

echo "  OK -- cert emitido."

# -- (5) Habilitar renovacion automatica (systemd timer custom) --------------
# NOTA (v7.6.2): cuando certbot viene del package manager del distro, este
# instala un certbot-renew.timer listo para usar. Como ahora viene de un
# venv, no trae unit de systemd, asi que creamos los nuestros. Esquema
# estandar Let's Encrypt: oneshot diario con jitter de 1h, --deploy-hook
# recarga nginx solo si efectivamente se renovo el cert.
echo
echo "[5/6] Habilitando renovacion automatica (systemd timer)..."

cat > /etc/systemd/system/certbot-renew.service <<EOF
[Unit]
Description=Renew Let's Encrypt certificates (certbot venv)
After=network-online.target

[Service]
Type=oneshot
ExecStart=/opt/certbot/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"
EOF

cat > /etc/systemd/system/certbot-renew.timer <<EOF
[Unit]
Description=Daily Let's Encrypt certificate renewal

[Timer]
OnCalendar=*-*-* 03:00:00
RandomizedDelaySec=1h
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now certbot-renew.timer
systemctl list-timers certbot-renew.timer --no-pager | head -3
echo "  OK -- el cert se renueva solo (timer diario, certbot decide si emitir)."

# -- (6) Test final ----------------------------------------------------------
echo
echo "[6/6] Test final..."
sleep 2
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://$DOMAIN/health" || echo "fail")
if [[ "$HTTP_CODE" == "200" ]]; then
    echo "  OK -- https://$DOMAIN/health responde 200."
else
    echo "  ADVERTENCIA -- https://$DOMAIN/health devolvio $HTTP_CODE"
    echo "  Verificar que uvicorn este corriendo: systemctl status maia"
fi

echo
echo "============================================================"
echo "  HTTPS listo: https://$DOMAIN"
echo "  Docs:        https://$DOMAIN/docs"
echo "  Health:      https://$DOMAIN/health"
echo "============================================================"
echo
echo "Cerrar el puerto 8000 al publico (opcional pero recomendado):"
echo "  AWS Console -> Security Groups -> remove TCP 8000 0.0.0.0/0"
echo "  Asi solo se accede via nginx en HTTPS."
echo
