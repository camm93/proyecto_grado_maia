"""
prediction_logger.py — Logger JSONL para predicciones T1.

Activación: variable de entorno T1_LOG_PREDICTIONS (default: false).
Valores que activan: "true", "1", "yes", "on" (case-insensitive).

Cuando está activo, cada predicción se escribe a:
    backend/logs/t1_predictions_YYYY-MM-DD.jsonl

Una línea por predicción, formato JSONL:
    {"ts": "2026-05-13T20:14:33.421Z",
     "model_id": "scibeto-es-t1",
     "input_hash": "a3f1...",            # SHA1 truncado, no almacena el texto
     "input_length": 234,
     "preprocess": {
        "regex_category": "METH",
        "regex_score": 6,
        "n_masks": 6,
        "matched_keywords": ["metodología", "muestra", ...]
     },
     "prediction": {
        "label": "METH", "confidence": 0.61,
        "alternatives": [{"label":"RES","score":0.21}, ...]
     },
     "elapsed_ms": 87}

NOTA: por defecto NO se almacena el texto del input (privacy-friendly).
Activar T1_LOG_FULL_TEXT=true SOLO en entornos de desarrollo para depuración.
"""

from __future__ import annotations
import os
import json
import hashlib
import logging
import datetime as dt
from pathlib import Path
from typing import Any

log = logging.getLogger("maia.predlog")

_TRUTHY = {"true", "1", "yes", "on", "y"}


def _truthy(env_var: str) -> bool:
    return os.getenv(env_var, "").strip().lower() in _TRUTHY


def is_logging_enabled() -> bool:
    """Toggle global T1: T1_LOG_PREDICTIONS controla si se loggean predicciones."""
    return _truthy("T1_LOG_PREDICTIONS")


def is_t2_logging_enabled() -> bool:
    """Toggle global T2: T2_LOG_PREDICTIONS controla si se loggean predicciones."""
    return _truthy("T2_LOG_PREDICTIONS")


def _log_full_text_enabled() -> bool:
    """Toggle adicional T1: T1_LOG_FULL_TEXT incluye el input completo (solo dev)."""
    return _truthy("T1_LOG_FULL_TEXT")


def _log_full_text_enabled_t2() -> bool:
    """Toggle adicional T2: T2_LOG_FULL_TEXT incluye el input completo (solo dev)."""
    return _truthy("T2_LOG_FULL_TEXT")


def _logs_dir() -> Path:
    """
    Directorio de logs. Default: backend/logs/ (relativo al package backend).
    Override: variable de entorno T1_LOG_DIR.
    """
    custom = os.getenv("T1_LOG_DIR")
    if custom:
        p = Path(custom)
    else:
        # backend/logs relativo a este archivo (backend/core/prediction_logger.py)
        p = Path(__file__).parent.parent / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _today_path(task: str = "t1") -> Path:
    """Path del archivo de log de hoy. `task` ∈ {'t1','t2'} para separar streams."""
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    return _logs_dir() / f"{task}_predictions_{today}.jsonl"


def _hash_text(text: str) -> str:
    """SHA1 de los primeros 64KB, truncado a 12 chars. Identificador estable."""
    h = hashlib.sha1(text[:65536].encode("utf-8")).hexdigest()
    return h[:12]


def log_t1_prediction(
    *,
    model_id: str,
    input_text: str,
    preprocess_info: dict[str, Any],
    prediction: dict[str, Any],
    elapsed_ms: int,
) -> None:
    """
    Loggea una predicción T1 si T1_LOG_PREDICTIONS está activado.

    No falla nunca: si hay un error de I/O, lo registra a stderr y sigue.
    """
    if not is_logging_enabled():
        return

    try:
        entry: dict[str, Any] = {
            "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "model_id": model_id,
            "input_hash": _hash_text(input_text),
            "input_length": len(input_text),
            "preprocess": {
                "regex_category": preprocess_info.get("regex_category"),
                "regex_score": preprocess_info.get("regex_score"),
                "n_masks": preprocess_info.get("neutralized_snippet", "").count("[MASK]"),
                "matched_keywords": preprocess_info.get("matched_keywords", []),
                "was_masked": preprocess_info.get("was_masked", False),
            },
            "prediction": {
                "label": prediction.get("label"),
                "confidence": prediction.get("confidence"),
                "alternatives": prediction.get("alternatives", []),
            },
            "elapsed_ms": elapsed_ms,
        }

        if _log_full_text_enabled():
            entry["input_text"] = input_text
            entry["neutralized_text"] = preprocess_info.get("neutralized_snippet", "")

        path = _today_path("t1")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        # No fallar el endpoint si el log falla
        log.warning(f"No se pudo escribir log de prediccion T1: {e}")


def log_t2_prediction(
    *,
    model_id: str,
    input_text: str,
    t1_context: dict[str, Any],
    prediction: dict[str, Any],
    elapsed_ms: int,
) -> None:
    """
    Loggea una predicción T2 si T2_LOG_PREDICTIONS está activado.

    A diferencia de T1, T2 no tiene preprocessing previo (texto crudo al modelo),
    así que el campo `preprocess` se reemplaza por `t1_context` que captura
    la metadata retórica del fragmento — útil para análisis de errores por
    zona del documento.

    No falla nunca: si hay un error de I/O, lo registra a stderr y sigue.
    """
    if not is_t2_logging_enabled():
        return

    try:
        entry: dict[str, Any] = {
            "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "model_id": model_id,
            "task": "T2",
            "input_hash": _hash_text(input_text),
            "input_length": len(input_text),
            "t1_context": {
                "t1_label": t1_context.get("t1_label"),
                "rhetorical_zone": t1_context.get("rhetorical_zone"),
                "relative_pos": t1_context.get("relative_pos"),
            },
            "prediction": {
                "label": prediction.get("label"),
                "confidence": prediction.get("confidence"),
                "p_contribution": prediction.get("p_contribution"),
                "threshold": prediction.get("threshold"),
                "all_probs": prediction.get("all_probs", {}),
            },
            "elapsed_ms": elapsed_ms,
        }

        if _log_full_text_enabled_t2():
            entry["input_text"] = input_text

        path = _today_path("t2")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        log.warning(f"No se pudo escribir log de prediccion T2: {e}")
