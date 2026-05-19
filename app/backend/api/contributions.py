"""Endpoint Tarea 2 — Detección de contribuciones.

Dispatch:
- model_id == "heuristic"                    : motor de reglas (siempre disponible)
- model_id ∈ {scibeto-es-t2, scibeto-es-t2-ctx}: SciBETO local
- model_id ∈ {llama-3.1-8b-t2, llama-3.1-8b-t2-fs8}: Llama local 4-bit
- model_id ∈ {gpt-4o-mini-t2, gpt-4o-mini-t2-fs8}: GPT API
- Cualquier otro                             : fallback transparente a heurística.

Todo lo que no sea heurístico está protegido con try/except + ModelNotAvailableError
y degrada a heurística con un mensaje claro en `model_name`.
"""

from __future__ import annotations

import time
import logging
from fastapi import APIRouter

from ..core.catalog import MODELS_T2, resolve_model
from ..core.heuristic import run_t2 as run_t2_heuristic
from ..core.t2_scibeto_service import run_t2_scibeto
from ..core.llm_service import run_t2_llm, LLM_DISPATCH
from ..model_loader import ModelNotAvailableError
from ..schemas import ContributionRequest, ContributionResponse, ContributionResult

log = logging.getLogger("maia.api.contributions")

router = APIRouter()


SCIBETO_T2_IDS = {"scibeto-es-t2", "scibeto-es-t2-ctx"}
LLM_T2_IDS     = set(LLM_DISPATCH.keys())  # {'llama-3.1-8b-t2', 'llama-3.1-8b-t2-fs8',
                                            #  'gpt-4o-mini-t2', 'gpt-4o-mini-t2-fs8'}


@router.post("/api/contributions", response_model=ContributionResponse,
             tags=["Tarea 2 -- Deteccion de Contribuciones"])
def contributions(req: ContributionRequest):
    """**Tarea 2** — Detección binaria de contribuciones científicas.

    Acepta 7 model_ids:
    - `heuristic`: reglas léxicas (baseline siempre disponible)
    - `scibeto-es-t2`: encoder fine-tuned, texto crudo (RECOMENDADO, F1=0.88)
    - `scibeto-es-t2-ctx`: encoder fine-tuned, con prefijo `[SECCION:X] [POS:Y]` (ablación)
    - `llama-3.1-8b-t2`: Llama 3.1 8B zero-shot (open-weight, GPU local)
    - `llama-3.1-8b-t2-fs8`: Llama 3.1 8B few-shot k=8
    - `gpt-4o-mini-t2`: GPT-4o-mini zero-shot (API)
    - `gpt-4o-mini-t2-fs8`: GPT-4o-mini few-shot k=8

    Fallback transparente: si el modelo no está available, devuelve heurística
    con un mensaje claro en `model_name` para que el frontend lo muestre.
    """
    info, mode, name = resolve_model(req.model_id, MODELS_T2)
    t0 = time.perf_counter()

    used_mode = mode
    used_name = name
    fragments_dicts = [f.model_dump() for f in req.fragments]

    # ── Dispatch ────────────────────────────────────────────────────────────
    if req.model_id in SCIBETO_T2_IDS and info["available"]:
        try:
            raw = run_t2_scibeto(fragments_dicts, model_id=req.model_id)
            used_mode = "encoder"
        except ModelNotAvailableError as e:
            log.warning(f"SciBETO T2 no disponible, fallback a heuristica: {e}")
            raw = run_t2_heuristic(fragments_dicts)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: pesos no encontrados]"
        except Exception as e:
            log.exception(f"Error en SciBETO T2, fallback a heuristica: {e}")
            raw = run_t2_heuristic(fragments_dicts)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: error de inferencia]"

    elif req.model_id in LLM_T2_IDS and info["available"]:
        try:
            raw = run_t2_llm(req.model_id, fragments_dicts)
            used_mode = info["family"]  # 'llm_open' o 'llm_api'
        except ModelNotAvailableError as e:
            log.warning(f"LLM {req.model_id} no disponible, fallback a heuristica: {e}")
            raw = run_t2_heuristic(fragments_dicts)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: LLM no disponible]"
        except Exception as e:
            log.exception(f"Error en LLM {req.model_id}, fallback a heuristica: {e}")
            raw = run_t2_heuristic(fragments_dicts)
            used_mode = "heuristic"
            used_name = info["name"] + " [fallback heuristico: error de LLM]"

    else:
        # heurístico default o modelo desconocido
        raw = run_t2_heuristic(fragments_dicts)

    elapsed = int((time.perf_counter() - t0) * 1000)

    detected = [r for r in raw if r["is_contribution"]]
    nc = len(detected)

    by_zone: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for r in detected:
        z = r["rhetorical_zone"]; by_zone[z] = by_zone.get(z, 0) + 1
        l = r["t1_label"];        by_label[l] = by_label.get(l, 0) + 1

    return ContributionResponse(
        model_id=req.model_id, model_name=used_name, mode=used_mode, elapsed_ms=elapsed,
        contributions=[ContributionResult(**{k: v for k, v in r.items() if not k.startswith("_")})
                       for r in raw],
        summary={
            "total_fragments": len(raw),
            "contributions_detected": nc,
            "contribution_rate": round(nc / max(len(raw), 1), 3),
            "by_rhetorical_zone": by_zone,
            "by_t1_label": by_label,
        },
    )
