#!/usr/bin/env bash
# download_models.sh -- Baja los pesos de los modelos T1 y T2 a /app/models/.
# ===============================================================================
# Soporta dos fuentes (HF Hub tiene prioridad si ambas estan definidas):
#
#   - HuggingFace Hub:  HF_TOKEN + HF_REPO_T1 / HF_REPO_T2 / HF_REPO_T2_CTX
#   - AWS S3:           S3_BUCKET + S3_PREFIX_T1 / S3_PREFIX_T2 / S3_PREFIX_T2_CTX
#
# Idempotente: si el modelo ya esta en disco (con model.safetensors), se salta.
# ===============================================================================
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/app/models}"
mkdir -p "$MODELS_DIR"

# Determinar la fuente de descarga
SOURCE="none"
if [[ -n "${HF_REPO_T1:-}${HF_REPO_T2:-}${HF_REPO_T2_CTX:-}" ]]; then
    SOURCE="hf"
elif [[ -n "${S3_BUCKET:-}" ]]; then
    SOURCE="s3"
fi

if [[ "$SOURCE" == "none" ]]; then
    echo "WARN:  No hay HF_REPO_* ni S3_BUCKET definidos. Skipping download."
    exit 0
fi

echo "> Fuente de descarga: $SOURCE"

# --- Helper para verificar si un modelo ya esta completo -----------------------
already_present() {
    local dir="$1"
    [[ -f "$dir/config.json" ]] && {
        [[ -f "$dir/model.safetensors" ]] || \
        [[ -f "$dir/pytorch_model.bin" ]]
    }
}

# --- HuggingFace Hub -----------------------------------------------------------
download_hf() {
    local repo="$1"
    local target_dir="$2"
    if already_present "$target_dir"; then
        echo "  skip:  $target_dir ya tiene pesos -- skip"
        return 0
    fi
    echo "  > HF Hub: $repo -> $target_dir"
    pip install --quiet huggingface_hub
    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='$repo',
    local_dir='$target_dir',
    token=os.environ.get('HF_TOKEN'),
    local_dir_use_symlinks=False,
)
"
}

# --- AWS S3 --------------------------------------------------------------------
download_s3() {
    local prefix="$1"
    local target_dir="$2"
    if already_present "$target_dir"; then
        echo "  skip:  $target_dir ya tiene pesos -- skip"
        return 0
    fi
    echo "  > S3: s3://$S3_BUCKET/$prefix -> $target_dir"
    command -v aws >/dev/null 2>&1 || pip install --quiet awscli
    aws s3 sync "s3://$S3_BUCKET/$prefix" "$target_dir/"
}

# --- Ejecutar descargas --------------------------------------------------------
if [[ "$SOURCE" == "hf" ]]; then
    [[ -n "${HF_REPO_T1:-}"     ]] && download_hf "$HF_REPO_T1"     "$MODELS_DIR/scibeto-es-t1"
    [[ -n "${HF_REPO_T2:-}"     ]] && download_hf "$HF_REPO_T2"     "$MODELS_DIR/scibeto-es-t2"
    [[ -n "${HF_REPO_T2_CTX:-}" ]] && download_hf "$HF_REPO_T2_CTX" "$MODELS_DIR/scibeto-es-t2-ctx"
elif [[ "$SOURCE" == "s3" ]]; then
    [[ -n "${S3_PREFIX_T1:-}"     ]] && download_s3 "$S3_PREFIX_T1"     "$MODELS_DIR/scibeto-es-t1"
    [[ -n "${S3_PREFIX_T2:-}"     ]] && download_s3 "$S3_PREFIX_T2"     "$MODELS_DIR/scibeto-es-t2"
    [[ -n "${S3_PREFIX_T2_CTX:-}" ]] && download_s3 "$S3_PREFIX_T2_CTX" "$MODELS_DIR/scibeto-es-t2-ctx"
fi

echo "[OK] Descarga completada"
