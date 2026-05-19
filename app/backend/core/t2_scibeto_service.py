"""
t2_scibeto_service.py — Orquestador T2 SciBETO (contribuciones)
================================================================
Encapsula el pipeline completo de inferencia T2 sobre los fragments
ya segmentados por T1:

    fragments[] → classify_batch(texts) → ensamblar resultado + log

A diferencia de `t1_scibeto_service.py`, NO hay preprocessing previo:
el notebook de entrenamiento (entrenamiento_t2.ipynb, celdas 12-14)
confirma que el modelo recibe el texto crudo del fragmento, sin
máscara dinámica, sin prefijos, sin metadatos retóricos como features.

Por eso este servicio es notablemente más corto que el de T1:
    text → tokenize → forward → softmax → label (no preprocess en el medio).

El contexto retórico (`t1_label`, `relative_pos`, `rhetorical_zone`)
se conserva en la salida porque el frontend y el reporte académico lo
muestran al usuario, pero no entra al modelo.

`run_t2_scibeto(fragments)` es el entry point principal. Lanza
ModelNotAvailableError si los pesos no están descargados.
"""

from __future__ import annotations
import os
import time
import logging
from typing import Any

from .prediction_logger import log_t2_prediction
from ..model_loader import get_t2_classifier, ModelNotAvailableError

log = logging.getLogger("maia.t2_scibeto")

# Threshold de decisión sobre la prob de ES_CONTRIBUCION (clase 1).
# El notebook usa argmax(logits) que es equivalente a threshold = 0.5.
# Exponemos T2_THRESHOLD por env var para calibración futura sin reentrenar:
# valores > 0.5 reducen falsos positivos (mayor precision), valores < 0.5
# reducen falsos negativos (mayor recall).
def _threshold() -> float:
    try:
        return float(os.getenv("T2_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


# Confidence threshold debajo del cual marcamos low_confidence=True.
# Para binaria, "el modelo está adivinando" es cuando la prob del label
# predicho está cerca de 0.5. Marcamos low_confidence si max(p) < 0.60.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Label names del modelo (deben matchear id2label del config.json T2)
LABEL_POSITIVE = "ES_CONTRIBUCION"
LABEL_NEGATIVE = "NO_ES_CONTRIBUCION"

# Etiquetas T1 válidas para el prefijo del modelo con contexto.
# Cualquier valor fuera de este conjunto se mapea a 'UNK' (consistente con
# `normalize_context()` del notebook entrenamiento_t2_con_contexto.ipynb).
_T1_VALID = {"INTRO", "BACK", "METH", "RES", "DISC", "LIM", "CONC"}


def _build_prefixed_text(frag: dict) -> str:
    """
    Construye el input con prefijo contextual para el modelo `scibeto-es-t2-ctx`.

    Formato (idéntico al notebook entrenamiento_t2_con_contexto.ipynb):
        "[SECCION:<t1_label>] [POS:<bin>] <texto>"
    Donde <bin> es relative_pos discretizado en 10 bins (0.0, 0.1, ..., 0.9).

    Para fragments sin t1_label o relative_pos (ej. heurística sin metadata
    completa), se usan defaults seguros: UNK + 0.5. El modelo aprende a manejar
    estos casos durante el train (la celda de carga de datos los normaliza
    igualmente).
    """
    t1 = str(frag.get("t1_label", "")).upper().strip()
    if t1 not in _T1_VALID:
        t1 = "UNK"
    try:
        pos = float(frag.get("relative_pos", 0.5))
    except (TypeError, ValueError):
        pos = 0.5
    pos = max(0.0, min(1.0, pos))
    pos_bin = max(0.0, min(0.9, round(pos * 10) / 10))
    text = str(frag.get("text", ""))
    return f"[SECCION:{t1}] [POS:{pos_bin:.1f}] {text}"


def _is_context_model(model_id: str) -> bool:
    """Convención: model_id terminado en '-ctx' usa prefijo contextual."""
    return model_id.endswith("-ctx")


def _build_rhetorical_context(
    *, is_contribution: bool, frag: dict, model_conf: float
) -> str:
    """
    Texto humano que se muestra en el frontend bajo la predicción.
    Solo se rellena para positivos (consistente con el heurístico).
    """
    if not is_contribution:
        return ""
    t1_name = frag.get("t1_label_name", frag.get("t1_label", ""))
    t1_conf = frag.get("t1_confidence", 0)
    zone = frag.get("rhetorical_zone", "")
    return (
        f"Contribucion detectada (conf. {model_conf*100:.0f}%) "
        f"en seccion '{t1_name}' "
        f"({t1_conf*100:.0f}% conf. retorica) | {zone}"
    )


def run_t2_scibeto(
    fragments: list[dict], model_id: str = "scibeto-es-t2"
) -> list[dict]:
    """
    Drop-in replacement para `run_t2` (heurística) cuando el modelo
    SciBETO T2 está disponible.

    Tokeniza y clasifica TODOS los fragments en BATCH (no uno por uno).
    En CPU el speedup vs loop secuencial es ~4x; en GPU ~20-50x.

    Lanza ModelNotAvailableError si los pesos no están descargados.
    El caller (api/contributions.py) debe manejarla con fallback a la
    heurística — patrón idéntico al de T1.
    """
    # Carga (con lru_cache) — falla rápido si los pesos no están
    classifier = get_t2_classifier(model_id)

    if not fragments:
        return []

    n = len(fragments)
    threshold = _threshold()
    use_context = _is_context_model(model_id)

    # ── Paso 1: construir el input según el tipo de modelo ──────────────────
    # - scibeto-es-t2     → texto crudo (notebook base)
    # - scibeto-es-t2-ctx → prefijo "[SECCION:X] [POS:Y] <texto>" (notebook +ctx)
    if use_context:
        texts = [_build_prefixed_text(f) for f in fragments]
    else:
        texts = [str(f.get("text", "")) for f in fragments]

    # ── Paso 2: clasificación en batch (donde se gana la latencia) ──────────
    t_inf0 = time.perf_counter()
    predictions = classifier.classify_batch(texts)
    t_inf_ms = int((time.perf_counter() - t_inf0) * 1000)

    log.info(
        f"T2 SciBETO inferencia: {n} fragments | "
        f"forward batch {t_inf_ms}ms | device={classifier.device} | "
        f"threshold={threshold} | ctx={use_context}"
    )

    # ── Paso 3: ensamblar resultado + log de cada predicción ────────────────
    results: list[dict] = []
    per_item_ms = t_inf_ms / max(n, 1)

    for idx, (frag, pred) in enumerate(zip(fragments, predictions)):
        # all_probs viene del wrapper genérico T1Classifier:
        # {"ES_CONTRIBUCION": 0.83, "NO_ES_CONTRIBUCION": 0.17}
        all_probs = pred["all_probs"]
        p_contribution = all_probs.get(LABEL_POSITIVE, 0.0)

        # Decisión basada en threshold sobre prob de ES_CONTRIBUCION,
        # equivalente al argmax cuando threshold=0.5 (default del notebook).
        is_c = p_contribution >= threshold

        if is_c:
            label = LABEL_POSITIVE
            confidence = p_contribution
        else:
            label = LABEL_NEGATIVE
            confidence = all_probs.get(LABEL_NEGATIVE, 1.0 - p_contribution)

        # Log individual (no-op si T2_LOG_PREDICTIONS=false)
        log_t2_prediction(
            model_id=model_id,
            input_text=texts[idx],
            t1_context={
                "t1_label": frag.get("t1_label", ""),
                "rhetorical_zone": frag.get("rhetorical_zone", ""),
                "relative_pos": frag.get("relative_pos"),
            },
            prediction={
                "label": label,
                "confidence": confidence,
                "p_contribution": p_contribution,
                "threshold": threshold,
                "all_probs": all_probs,
            },
            elapsed_ms=int(per_item_ms),
        )

        t1_name = frag.get("t1_label_name", frag.get("t1_label", ""))
        t1_conf = frag.get("t1_confidence", 0)
        zone = frag.get("rhetorical_zone", "")

        results.append({
            "index": frag.get("index", idx),
            "is_contribution": is_c,
            "confidence": round(confidence, 4),
            "label": label,
            "t1_label": frag.get("t1_label", ""),
            "t1_label_name": t1_name,
            "t1_color": frag.get("t1_color", "#9CA3AF"),
            "t1_confidence": t1_conf,
            "rhetorical_zone": zone,
            "rhetorical_context": _build_rhetorical_context(
                is_contribution=is_c, frag=frag, model_conf=confidence,
            ),
            "char_start": frag.get("char_start"),
            "char_end": frag.get("char_end"),
            "word_count": frag.get("word_count"),
            "relative_pos": frag.get("relative_pos"),
        })

    return results
