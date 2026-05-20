#!/usr/bin/env bash
# start.sh - Arranque del backend MAIA (Linux/macOS).
# ============================================================================
# Hace, en este orden:
#   (pre) Instala antiword si falta (parser de .doc legacy, build ~30s)
#         - Linux:    sudo scripts/install_antiword.sh (idempotente)
#         - Otros:    warn y continua sin soporte .doc
#   (a) pip install -r backend/requirements.txt   (idempotente)
#   (b) Sincronizacion + verificacion de pesos via model_sync
#       - 1er arranque: descarga 3 ZIPs desde Drive (~3.9 GB, ~5-8 min)
#       - Re-arranques: solo verifica SHAs (~5-10 s)
#   (c) uvicorn backend.main:app --host 0.0.0.0 --port 8000
#
# Uso:
#   ./start.sh                       Arranque normal (dev local o produccion)
#   ./start.sh --port 8080           Cambia el puerto (default 8000)
#   ./start.sh --no-reload           Sin --reload (recomendado en produccion)
#   ./start.sh --skip-install        Omite el pip install (re-arranques rapidos)
#   ./start.sh --skip-sync           Omite model_sync (asume pesos ya OK)
#   ./start.sh --skip-antiword       Omite check/install de antiword
#   ./start.sh --verify-only         Solo verifica integridad y sale
#   ./start.sh --help                Esta ayuda
#
# Variables de entorno relevantes (ver backend/utils/model_sync.py):
#   MODELS_DIR              default ./models
#   MANIFEST_PATH           default ./release_manifest.json
#   DOWNLOAD_FAIL_POLICY    abort | warn (default abort)
#   T1_PRELOAD              true | false (default true)
#   T2_PRELOAD              true | false (default true)
#   T1_PRELOAD_ID           default scibeto-es-t1
#   T2_PRELOAD_ID           default scibeto-es-t2
# ============================================================================
set -euo pipefail

# -- Defaults / parseo de argumentos -----------------------------------------
PORT=8000
DO_INSTALL=true
DO_SYNC=true
DO_RELOAD=true
DO_ANTIWORD=true
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)          PORT="$2"; shift 2 ;;
        --no-reload)     DO_RELOAD=false; shift ;;
        --skip-install)  DO_INSTALL=false; shift ;;
        --skip-sync)     DO_SYNC=false; shift ;;
        --skip-antiword) DO_ANTIWORD=false; shift ;;
        --verify-only)   VERIFY_ONLY=true; shift ;;
        --help|-h)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *)
            echo "Argumento desconocido: $1 -- ver --help"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

echo "============================================================"
echo "  MAIA -- Analisis Retorico (v7.8.2)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# -- (pre) antiword (parser de .doc legacy) ----------------------------------
# antiword es necesario para el endpoint POST /api/extract-doc que convierte
# archivos .doc (Word 97-2003 binario) a texto. NO esta en repos oficiales
# de Amazon Linux 2023, asi que en primer arranque se compila via
# scripts/install_antiword.sh (idempotente: si ya esta, sale OK sin trabajo).
#
# Manejo de fallos: si la instalacion no funciona (no hay sudo, no hay
# internet, etc.), imprimimos warning y CONTINUAMOS — uvicorn arranca
# igual, solo la feature .doc va a devolver 503 al usuario hasta que
# antiword se instale manualmente. El resto de MAIA (.docx, .pdf, .txt)
# funciona sin antiword.
if [[ "$DO_ANTIWORD" == "true" ]]; then
    echo
    echo "[pre] Verificando antiword (soporte de .doc legacy)..."
    if command -v antiword >/dev/null 2>&1; then
        echo "  OK -- antiword ya instalado en $(command -v antiword)"
    else
        OS_NAME=$(uname -s)
        if [[ "$OS_NAME" == "Linux" ]]; then
            if [[ -x scripts/install_antiword.sh ]]; then
                echo "  antiword no detectado. Intentando instalar (requiere sudo)..."
                # Intentar el install. Si falla por cualquier razon (sudo
                # negado, network, build error), capturamos y seguimos.
                if sudo ./scripts/install_antiword.sh; then
                    echo "  OK -- antiword instalado."
                else
                    echo
                    echo "  ADVERTENCIA: La instalacion automatica de antiword fallo."
                    echo "  El servidor va a arrancar igual, pero la feature de"
                    echo "  subir archivos .doc va a devolver 503 hasta que se"
                    echo "  instale manualmente con:"
                    echo "      sudo ./scripts/install_antiword.sh"
                    echo "  Los formatos .docx, .pdf y .txt funcionan sin antiword."
                    echo
                fi
            else
                echo "  ADVERTENCIA: scripts/install_antiword.sh no encontrado"
                echo "  o no ejecutable. Soporte .doc deshabilitado."
            fi
        else
            echo "  Sistema no-Linux ($OS_NAME) — instalacion automatica saltada."
            echo "  Para soporte de archivos .doc legacy en este sistema:"
            echo "      macOS:        brew install antiword"
            echo "      Ubuntu/Deb:   sudo apt install -y antiword"
            echo "      Otros:        ver scripts/install_antiword.sh"
        fi
    fi
else
    echo
    echo "[pre] antiword check omitido (--skip-antiword). Soporte .doc puede"
    echo "      no estar disponible si no se instalo manualmente antes."
fi

# -- (a) Dependencias --------------------------------------------------------
if [[ "$DO_INSTALL" == "true" ]]; then
    echo
    echo "[a/c] Instalando/verificando dependencias..."

    if [[ ! -d "venv" ]]; then
        echo "  Creando entorno virtual ./venv ..."

        # Buscar el mejor Python disponible. Requisito: >=3.10 (varios paquetes
        # como gdown>=6, transformers nuevos, etc. lo requieren).
        # En Amazon Linux 2023 el python3 default es 3.9 — instalar python3.11
        # con: sudo dnf install -y python3.11 python3.11-pip python3.11-devel
        PYTHON_BIN=""
        for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then
                version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
                major=$(echo "$version" | cut -d. -f1)
                minor=$(echo "$version" | cut -d. -f2)
                if [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 10 ]]; }; then
                    PYTHON_BIN="$candidate"
                    echo "  Usando $candidate (version $version)"
                    break
                fi
            fi
        done

        if [[ -z "$PYTHON_BIN" ]]; then
            echo "  ERROR: No se encontro Python >=3.10 instalado."
            echo "  En Amazon Linux 2023 instalar con:"
            echo "      sudo dnf install -y python3.11 python3.11-pip python3.11-devel"
            echo "  En Ubuntu/Debian:"
            echo "      sudo apt install -y python3.11 python3.11-venv python3.11-dev"
            echo "  En macOS:"
            echo "      brew install python@3.11"
            exit 1
        fi

        "$PYTHON_BIN" -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate

    # Upgrade pip via `python -m pip` (no `pip install --upgrade pip` directo,
    # porque pip se rehusa a auto-modificarse cuando lo invoca el .exe en
    # Windows; en Linux funciona de las dos formas, pero esto es portable).
    # El `|| true` evita abortar si el upgrade falla — un pip viejo no es
    # bloqueante para el resto del install.
    python -m pip install -q --upgrade pip || true
    pip install -q -r backend/requirements.txt
    echo "  OK -- Dependencias listas"
else
    [[ -d "venv" ]] && source venv/bin/activate
    echo "[a/c] pip install omitido (--skip-install)"
fi

# -- (b) Sincronizacion de modelos -------------------------------------------
export MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
export MANIFEST_PATH="${MANIFEST_PATH:-$(pwd)/release_manifest.json}"

if [[ "$VERIFY_ONLY" == "true" ]]; then
    echo
    echo "[verify-only] Verificando integridad SHA256 de pesos en $MODELS_DIR"
    python -m backend.utils.model_sync verify
    echo "  OK -- Verificacion completada"
    exit 0
fi

if [[ "$DO_SYNC" == "true" ]]; then
    echo
    echo "[b/c] Sincronizando pesos de modelos..."
    echo "  MODELS_DIR=$MODELS_DIR"
    echo "  MANIFEST_PATH=$MANIFEST_PATH"
    echo "  DOWNLOAD_FAIL_POLICY=${DOWNLOAD_FAIL_POLICY:-abort}"
    echo "  (1er arranque: ~5-8 min descargando ~3.9 GB desde Drive)"
    echo "  (re-arranques: ~5-10 s, solo verifica SHAs)"
    python -m backend.utils.model_sync ensure_all
else
    echo "[b/c] model_sync omitido (--skip-sync). Asumiendo pesos ya OK."
fi

# -- (c) uvicorn -------------------------------------------------------------
echo
echo "[c/c] Arrancando uvicorn en http://0.0.0.0:$PORT"
echo "  Docs:   http://localhost:$PORT/docs"
echo "  Health: http://localhost:$PORT/health"
echo "  Ctrl+C para detener."
echo "============================================================"

UVICORN_ARGS=(backend.main:app --host 0.0.0.0 --port "$PORT")
if [[ "$DO_RELOAD" == "true" ]]; then
    UVICORN_ARGS+=(--reload)
fi

exec uvicorn "${UVICORN_ARGS[@]}"
