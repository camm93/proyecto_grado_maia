"""Endpoint Tarea 1 — Segmentación retórica.

Dispatch:
- model_id == "heuristic": motor de reglas (siempre disponible)
- model_id == "scibeto-es-t1": SciBETO V5 fine-tuned si los pesos están en MODELS_DIR;
                            fallback a heurística si no.
- model_id ∈ {"llama-3.1-8b-t1", "gpt-4o-mini-t1"}: LLM dispatch via llm_service.run_t1_llm
                            (siempre few-shot k=14 — mejor métrica del notebook v2).
                            Llama requiere GPU + HF_TOKEN; GPT requiere OPENAI_API_KEY.
                            Cualquier error de credenciales/recursos → fallback heurístico.
"""

from __future__ import annotations

import time
import logging
from fastapi import APIRouter

from ..core.catalog import MODELS_T1, resolve_model
from ..core.heuristic import run_t1 as run_t1_heuristic, split_paragraphs
from ..core.lexicon import T1_NAMES, T1_COLORS
from ..core.t1_scibeto_service import run_t1_scibeto, _rhetorical_zone, LOW_CONFIDENCE_THRESHOLD
from ..core.llm_service import LLM_DISPATCH_T1, run_t1_llm
from ..model_loader import ModelNotAvailableError
from ..schemas import SegmentRequest, SegmentResponse, SegmentResult

log = logging.getLogger("maia.api.segment")

router = APIRouter()


def _assemble_t1_segments_from_llm(text: str, llm_results: list[dict]) -> list[dict]:
    """Combina párrafos crudos + predicciones del LLM en el schema T1
    estándar. Equivalente al ensamble que hace t1_scibeto_service.run_t1_scibeto,
    pero usando los resultados de run_t1_llm().
    """
    paragraphs = split_paragraphs(text)
    n = len(paragraphs)
    if n == 0:
        return []

    segments = []
    char_offset = 0
    for idx, (para, pred) in enumerate(zip(paragraphs, llm_results)):
        pos = idx / max(n - 1, 1)
        label = pred["label"]
        conf = pred["confidence"]

        cs = text.find(para, char_offset)
        char_start = cs if cs >= 0 else char_offset
        char_end = char_start + len(para)
        char_offset = char_end

        segments.append({
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
            "low_confidence": conf < LOW_CONFIDENCE_THRESHOLD,
        })
    return segments


@router.post("/api/segment", response_model=SegmentResponse,
             tags=["Tarea 1 -- Segmentacion Retorica"])
def segment(req: SegmentRequest):
    """**Tarea 1** — Segmentación retórica del texto académico.

    Divide el texto en párrafos y asigna a cada uno una de las etiquetas
    retóricas. Soporta 4 modelos T1:
      - `heuristic`: reglas léxico-posicionales (8 clases con CONTR)
      - `scibeto-es-t1`: SciBETO-large fine-tuned (7 clases sin CONTR)
      - `llama-3.1-8b-t1`: Llama 3.1 8B few-shot k=14
      - `gpt-4o-mini-t1`: GPT-4o-mini few-shot k=14 + JSON mode

    Los segmentos devueltos incluyen metadata de posición (char_start,
    char_end, relative_pos, rhetorical_zone) que debe pasarse como input
    a `/api/contributions`.
    """
    info, mode, name = resolve_model(req.model_id, MODELS_T1)
    t0 = time.perf_counter()

    used_mode = mode
    used_name = name

    # Branch 1: SciBETO encoder T1
    if req.model_id == "scibeto-es-t1" and info["available"]:
        try:
            raw = run_t1_scibeto(req.text, model_id=req.model_id)
            used_mode = "encoder"
        except ModelNotAvailableError as e:
            log.warning(f"SciBETO no disponible, fallback a heuristica: {e}")
            raw = run_t1_heuristic(req.text)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: pesos no encontrados]"
        except Exception as e:
            log.exception(f"Error en SciBETO V5, fallback a heuristica: {e}")
            raw = run_t1_heuristic(req.text)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: error de inferencia]"

    # Branch 2: LLM T1 (Llama o GPT, ambos few-shot k=14)
    elif req.model_id in LLM_DISPATCH_T1 and info["available"]:
        try:
            paragraphs = split_paragraphs(req.text)
            llm_results = run_t1_llm(req.model_id, paragraphs)
            raw = _assemble_t1_segments_from_llm(req.text, llm_results)
            used_mode = info["family"]  # "llm_open" o "llm_api"
        except ModelNotAvailableError as e:
            log.warning(f"LLM {req.model_id} no disponible, fallback heuristica: {e}")
            raw = run_t1_heuristic(req.text)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: credenciales/GPU faltantes]"
        except Exception as e:
            log.exception(f"Error en LLM {req.model_id}, fallback heuristica: {e}")
            raw = run_t1_heuristic(req.text)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: error de inferencia]"

    # Branch 3: heuristic (default) o cualquier otro ID no integrado
    else:
        raw = run_t1_heuristic(req.text)

    elapsed = int((time.perf_counter() - t0) * 1000)

    dist: dict[str, int] = {}
    confs: list[float] = []
    for s in raw:
        dist[s["t1_label"]] = dist.get(s["t1_label"], 0) + 1
        confs.append(s["t1_confidence"])

    return SegmentResponse(
        model_id=req.model_id, model_name=used_name, mode=used_mode, elapsed_ms=elapsed,
        segments=[SegmentResult(**s) for s in raw],
        summary={
            "total_segments": len(raw),
            "t1_distribution": dist,
            "avg_confidence": round(sum(confs) / max(len(confs), 1), 3),
            "low_confidence_count": sum(1 for s in raw if s["low_confidence"]),
        },
    )
