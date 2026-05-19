#!/usr/bin/env bash
# entrypoint.sh - Pre-arranque del contenedor MAIA.
# ============================================================================
# Camino principal (v5+): sincroniza modelos desde Google Drive con
# `model_sync ensure_all`. Lee la lista, URLs y hashes desde
# release_manifest.json. Verificacion dual de integridad:
#   - SHA256 del ZIP completo (al bajar)
#   - SHA256 por archivo `required: true` (al arrancar, cada vez)
#
# Variables relevantes (ver backend/utils/model_sync.py):
#   MODELS_DIR             /app/models
#   MANIFEST_PATH          /app/release_manifest.json
#   DOWNLOAD_FAIL_POLICY   abort (default) | warn
#   MIN_FREE_DISK_GB       10 (default)
#
# Camino legacy: si DOWNLOAD_MODELS_ON_START=true se invoca
# scripts/download_models.sh con HF Hub o S3 como fuente. Util para
# entornos donde Drive no es accesible. DOWNLOAD_MODELS_ON_START tiene
# prioridad sobre el camino Drive.
# ============================================================================
set -euo pipefail

export MODELS_DIR="${MODELS_DIR:-/app/models}"
export MANIFEST_PATH="${MANIFEST_PATH:-/app/release_manifest.json}"

if [[ "${DOWNLOAD_MODELS_ON_START:-false}" == "true" ]]; then
    echo "=== [LEGACY] Descarga via HF Hub/S3 (DOWNLOAD_MODELS_ON_START=true) ==="
    /app/scripts/download_models.sh
else
    echo "=== Sync de modelos via Google Drive (model_sync ensure_all) ==="
    echo "  MODELS_DIR=$MODELS_DIR"
    echo "  MANIFEST_PATH=$MANIFEST_PATH"
    echo "  DOWNLOAD_FAIL_POLICY=${DOWNLOAD_FAIL_POLICY:-abort}"
    python -m backend.utils.model_sync ensure_all
fi

# Reporte de modelos detectados (informativo)
if [[ -d "$MODELS_DIR" ]]; then
    echo "Modelos detectados en $MODELS_DIR:"
    for d in "$MODELS_DIR"/*/; do
        [[ -d "$d" ]] || continue
        if [[ -f "$d/model.safetensors" || -f "$d/pytorch_model.bin" ]]; then
            echo "  [OK] $(basename "$d")"
        fi
    done
fi

echo ">> Arrancando uvicorn..."
exec "$@"
