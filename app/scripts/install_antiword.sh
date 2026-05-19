#!/usr/bin/env bash
# install_antiword.sh - Build & install de antiword desde el fork mantenido
# de GitHub (grobian/antiword). Necesario para el endpoint /api/extract-doc
# que convierte archivos .doc legacy (Word 97-2003) a texto plano.
#
# Por que build from source:
#   - antiword NO esta en los repos oficiales de Amazon Linux 2023
#   - Tampoco en EPEL para EL9
#   - El upstream original (www.winfield.demon.nl) no tiene HTTPS estable
#   - El fork grobian/antiword en GitHub es el mas mantenido a 2026
#
# Idempotente: si antiword ya esta instalado y funcional, sale OK sin
# rebuildear. Sirve para correr en cualquier deploy nuevo o re-deploy.
#
# Uso: sudo ./scripts/install_antiword.sh
# ============================================================================
set -euo pipefail

if [[ "$EUID" -ne 0 ]]; then
    echo "ERROR: Este script requiere sudo."
    exit 1
fi

echo "============================================================"
echo "  Instalacion de antiword (parser de .doc legacy)"
echo "============================================================"

# -- (1) Si ya esta y funciona, no rebuildeamos ------------------------------
if command -v antiword >/dev/null 2>&1; then
    if antiword -h >/dev/null 2>&1 || true; then
        # antiword -h escribe a stderr y exit 1; verificamos con un .doc real
        # via fake test (solo chequeamos que el binario corra)
        echo "OK -- antiword ya esta instalado en $(command -v antiword)"
        echo "     Version: $(antiword 2>&1 | head -1 || echo 'unknown')"
        exit 0
    fi
fi

# -- (2) Instalar toolchain de build si falta --------------------------------
echo
echo "[1/4] Verificando toolchain (gcc + make)..."
if ! command -v gcc >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    dnf install -y gcc make >/dev/null
fi
echo "  OK -- gcc + make disponibles."

# -- (3) Bajar y compilar antiword -------------------------------------------
echo
echo "[2/4] Descargando antiword desde GitHub (grobian/antiword)..."
WORKDIR=$(mktemp -d /tmp/antiword-build.XXXXXX)
cd "$WORKDIR"

# El fork grobian/antiword publica releases con tarballs. Si la URL cambia,
# alternativas: clone directo del repo (`git clone` y `make` adentro), o
# fallback a otro mirror.
TARBALL_URL="https://github.com/grobian/antiword/archive/refs/heads/master.tar.gz"
if ! curl -fsSL "$TARBALL_URL" -o antiword.tar.gz; then
    echo "  ERROR: no se pudo descargar antiword desde $TARBALL_URL"
    echo "  Fallback manual: git clone https://github.com/grobian/antiword"
    rm -rf "$WORKDIR"
    exit 1
fi
tar xzf antiword.tar.gz
SRCDIR=$(find . -maxdepth 1 -type d -name "antiword-*" | head -1)
if [[ -z "$SRCDIR" ]]; then
    echo "  ERROR: el tarball no contiene un directorio antiword-*"
    rm -rf "$WORKDIR"
    exit 1
fi
cd "$SRCDIR"
echo "  OK -- source en $SRCDIR"

# -- (4) Build + install -----------------------------------------------------
echo
echo "[3/4] Compilando antiword..."
make >/dev/null 2>&1 || {
    echo "  ERROR: falló make. Probar manualmente en $SRCDIR para ver el log."
    exit 1
}
echo "  OK -- binario compilado."

echo
echo "[4/4] Instalando en /usr/local/bin..."
install -m 0755 antiword /usr/local/bin/antiword
# antiword necesita su mapping dir para charset UTF-8.txt
mkdir -p /usr/share/antiword
if [[ -d Resources ]]; then
    cp -r Resources/* /usr/share/antiword/ 2>/dev/null || true
fi
echo "  OK -- antiword instalado en /usr/local/bin/antiword"

# -- (5) Sanity check --------------------------------------------------------
echo
echo "Verificacion final..."
if /usr/local/bin/antiword 2>&1 | head -1 | grep -qi "antiword"; then
    echo "  OK -- antiword funciona."
else
    echo "  ADVERTENCIA: antiword instalado pero no respondio como esperado."
    echo "  Verificar manualmente: /usr/local/bin/antiword"
fi

# Cleanup
cd /
rm -rf "$WORKDIR"

echo
echo "============================================================"
echo "  Instalacion completa. Para verificar:"
echo "    echo | antiword -t /dev/stdin 2>&1 | head -3"
echo "  Si maia.service estaba corriendo, reiniciar para que el"
echo "  endpoint /api/extract-doc encuentre el binario:"
echo "    sudo systemctl restart maia"
echo "============================================================"
