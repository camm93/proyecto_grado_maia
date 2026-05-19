"""
t1_scibeto_service.py — Orquestador T1 SciBETO V5
====================================================
Encapsula el pipeline completo de inferencia T1 sobre documentos:

    text → split_paragraphs → preprocess (batch) → classify (batch) → log

`run_t1_scibeto(text)` es el entry point principal: procesa un documento
completo en BATCH (preprocess en CPU, forward en GPU/CPU con padding). Es
mucho más rápido que el loop secuencial por párrafo.

`classify_paragraph_with_scibeto(...)` se conserva para tests unitarios
y para casos donde el caller quiere clasificar un único párrafo aislado.
"""

from __future__ import annotations
import time
import logging
from typing import Any

from .heuristic import split_paragraphs
from .lexicon import T1_NAMES, T1_COLORS
from .t1_preprocess import preprocess_for_t1
from .prediction_logger import log_t1_prediction
from ..model_loader import get_t1_classifier, ModelNotAvailableError

log = logging.getLogger("maia.t1_scibeto")

# Confidence threshold debajo del cual marcamos low_confidence=True.
# Para SciBETO V5, F1 sobre gold es 0.40, y la mayoría de predicciones
# correctas tienen prob > 0.50. Conf < 0.35 = el modelo está adivinando.
LOW_CONFIDENCE_THRESHOLD = 0.35


def _rhetorical_zone(pos: float) -> str:
    if pos <= 0.20:
        return "Inicio (0-20%)"
    if pos >= 0.80:
        return "Cierre (80-100%)"
    return "Desarrollo (20-80%)"


def classify_paragraph_with_scibeto(
    *, model_id: str, paragraph: str, classify_fn
) -> dict[str, Any]:
    """
    Clasifica un párrafo individual.

    Reproduce el pipeline V5 byte a byte:
    1. Pasar el párrafo crudo por el regex extractor → matched_keywords
    2. Aplicar máscara dinámica con esos matched_keywords
    3. Pasar el neutralized_snippet por SciBETO V5 → label + confidence + alternatives

    NO retorna metadata posicional (char_start, char_end, etc.) — eso lo agrega
    el caller que tiene el contexto del documento.
    """
    t0 = time.perf_counter()

    # Preprocesar (idéntico a V5 en train)
    pp = preprocess_for_t1(paragraph)

    # Clasificar con SciBETO
    pred = classify_fn(pp["neutralized_snippet"])

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Loggear (no-op si T1_LOG_PREDICTIONS está off)
    log_t1_prediction(
        model_id=model_id,
        input_text=paragraph,
        preprocess_info=pp,
        prediction=pred,
        elapsed_ms=elapsed_ms,
    )

    return {
        "label": pred["label"],
        "confidence": pred["confidence"],
        "alternatives": pred["alternatives"],
        "preprocess": {
            "regex_category": pp["regex_category"],
            "regex_score": pp["regex_score"],
            "was_masked": pp["was_masked"],
            "n_keywords_masked": len(pp["matched_keywords"]),
        },
        "elapsed_ms": elapsed_ms,
    }


def run_t1_scibeto(text: str, model_id: str = "scibeto-es-t1") -> list[dict]:
    """
    Drop-in replacement para `run_t1` (heurística) cuando el modelo
    SciBETO V5 está disponible.

    Divide el texto en párrafos, los preprocesa, y los clasifica todos
    en un BATCH al modelo (no uno por uno). Esto es crítico para latencia:
    en CPU, batching reduce ~4x el tiempo total; en GPU, ~20-50x.

    Lanza ModelNotAvailableError si los pesos no están descargados.
    El caller debe manejarla con fallback a la heurística.
    """
    # Carga (con caché lru_cache) — falla rápido si los pesos no están
    classifier = get_t1_classifier(model_id)

    paragraphs = split_paragraphs(text)
    n = len(paragraphs)
    if n == 0:
        return []

    # ── Paso 1: preprocess de TODOS los párrafos (CPU, barato) ──────────────
    t_pre0 = time.perf_counter()
    preprocessed = [preprocess_for_t1(p) for p in paragraphs]
    neutralized_texts = [pp["neutralized_snippet"] for pp in preprocessed]
    t_pre_ms = int((time.perf_counter() - t_pre0) * 1000)

    # ── Paso 2: clasificación en batch (donde se gana la latencia) ──────────
    t_inf0 = time.perf_counter()
    predictions = classifier.classify_batch(neutralized_texts)
    t_inf_ms = int((time.perf_counter() - t_inf0) * 1000)

    log.info(
        f"T1 SciBETO inferencia: {n} parrafos | preprocess {t_pre_ms}ms | "
        f"forward batch {t_inf_ms}ms | device={classifier.device}"
    )

    # ── Paso 3: ensamblar resultado + log de cada predicción ────────────────
    results: list[dict] = []
    char_offset = 0
    per_item_ms = t_inf_ms / max(n, 1)  # tiempo promedio prorrateado por item

    for idx, (para, pp, pred) in enumerate(zip(paragraphs, preprocessed, predictions)):
        pos = idx / max(n - 1, 1)
        label = pred["label"]
        conf = pred["confidence"]
        low_conf = conf < LOW_CONFIDENCE_THRESHOLD

        # Log individual de cada predicción (no-op si T1_LOG_PREDICTIONS=false)
        log_t1_prediction(
            model_id=model_id,
            input_text=para,
            preprocess_info=pp,
            prediction=pred,
            elapsed_ms=int(per_item_ms),
        )

        # Metadata posicional
        cs = text.find(para, char_offset)
        char_start = cs if cs >= 0 else char_offset
        char_end = char_start + len(para)
        char_offset = char_end

        results.append({
            "index": idx,
            "text": para,
            "char_start": char_start,
            "char_end": char_end,
            "word_count": len(para.split()),
            "relative_pos": round(pos, 4),
            "rhetorical_zone": _rhetorical_zone(pos),
            "t1_label": label,
            "t1_label_name": T1_NAMES.get(label, label),
            "t1_color": T1_COLORS.get(label, "#9CA3AF"),
            "t1_confidence": conf,
            "low_confidence": low_conf,
        })

    return results
