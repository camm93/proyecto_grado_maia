"""Endpoints de sistema: health, catálogo de modelos, model info, lexicón compartido."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException

from ..core.catalog import MODELS_T1, MODELS_T2, refresh_catalog
from ..core import lexicon
from ..core.prediction_logger import is_logging_enabled, is_t2_logging_enabled
from ..model_loader import get_model_metadata, list_available_models

router = APIRouter()


@router.get("/health", tags=["Sistema"])
def health():
    refresh_catalog()
    return {
        "status": "ok",
        "version": "2.1.0",
        "endpoints": ["/api/segment", "/api/contributions", "/api/lexicon",
                      "/api/models", "/api/models/{model_id}/info",
                      "/api/few_shots"],
        "models_t1": [m["id"] for m in MODELS_T1 if m["available"]],
        "models_t2": [m["id"] for m in MODELS_T2 if m["available"]],
        "weights_on_disk": list_available_models(),
        "t1_prediction_logging": is_logging_enabled(),
        "t2_prediction_logging": is_t2_logging_enabled(),
    }


@router.get("/api/models", tags=["Modelos"])
def get_models(task: Optional[str] = None):
    """Lista los modelos disponibles.
    - `?task=T1` → solo modelos de segmentación retórica
    - `?task=T2` → solo modelos de detección de contribuciones
    - Sin parámetro → todos los modelos de ambas tareas

    Recomputa `available` en cada request (refresh_catalog) para que los
    cambios en .env / pesos descargados se reflejen sin reiniciar uvicorn.
    """
    refresh_catalog()
    if task == "T1":
        return {"task": "T1", "models": MODELS_T1}
    if task == "T2":
        return {"task": "T2", "models": MODELS_T2}
    return {"T1": MODELS_T1, "T2": MODELS_T2}


@router.get("/api/models/{model_id}/info", tags=["Modelos"])
def get_model_info(model_id: str):
    """Metadata detallada de un modelo: arquitectura, clases, métricas reportadas.

    Lee `config.json` y opcionalmente `eval_metrics.json` desde MODELS_DIR.
    Solo disponible para modelos con pesos descargados.
    """
    refresh_catalog()
    all_models = {m["id"]: m for m in (MODELS_T1 + MODELS_T2)}
    if model_id not in all_models:
        raise HTTPException(404, f"Modelo '{model_id}' no esta en el catalogo.")

    catalog_entry = all_models[model_id]
    metadata = get_model_metadata(model_id)

    # Caveat de inferencia: el F1 reportado es sobre snippets test-like (250-1000w);
    # el demostrador procesa párrafos arbitrarios que pueden ser más cortos,
    # lo que puede degradar el desempeño en producción.
    if metadata.get("available") and catalog_entry.get("task") == "T1":
        metadata["inference_caveat"] = (
            "El F1 reportado se midio sobre snippets de 250-1000 palabras "
            "con preprocesamiento de mascara dinamica. El demostrador divide "
            "el texto en parrafos arbitrarios y puede ver degradacion en "
            "parrafos muy cortos (<50 palabras) o muy largos (>1000 palabras)."
        )

    return {
        "model_id": model_id,
        "catalog": catalog_entry,
        "metadata": metadata,
    }


@router.get("/api/lexicon", tags=["Modelos"])
def get_lexicon():
    """Lexicón compartido (regex strings) usado por el motor heurístico.

    El frontend lo consume al arrancar y compila los patrones con flag `gi`,
    eliminando la duplicación de regex entre Python y JavaScript.
    """
    return lexicon.as_payload()


@router.get("/api/few_shots", tags=["Modelos"])
def get_few_shots():
    """Devuelve los ejemplos few-shot publicados (T1 k=14 y T2 k=8) para
    consulta desde el frontend.

    Single source of truth: los textos viven en backend/core/llm_service.py
    (FEW_SHOT_EXAMPLES_T1 y FEW_SHOT_EXAMPLES). Los SHA256 publicados en el
    release_manifest.json se generan a partir de estas mismas listas, asi
    que servirlos por API garantiza que el modal del frontend muestra
    EXACTAMENTE los mismos textos auditados.
    """
    from ..core.llm_service import FEW_SHOT_EXAMPLES_T1, FEW_SHOT_EXAMPLES
    return {
        "T1": {
            "k": len(FEW_SHOT_EXAMPLES_T1),
            "task": "Segmentacion retorica (7 clases)",
            "balance": "2 ejemplos por clase x 7 clases (INTRO/BACK/METH/RES/DISC/LIM/CONC)",
            "order": "fijo (no shuffle) para SHA reproducible",
            "sha256": "7d154afd9c19699d1c637f2b98028eff964fbb8e4d3bc4f06005d6093387cec0",
            "examples": [{"label": lbl, "text": txt}
                         for lbl, txt in FEW_SHOT_EXAMPLES_T1],
        },
        "T2": {
            "k": len(FEW_SHOT_EXAMPLES),
            "task": "Deteccion binaria de contribuciones",
            "balance": "4 positivos (SI) + 4 negativos (NO), orden alternado",
            "order": "alternado (SI, NO, SI, NO, ...) para neutralizar sesgo posicional",
            "sha256": "4ccc1238b0afcc0b3885d818d911881ce8f372d9c17ed6bc43b918d14e20d509",
            "examples": [{"label": lbl, "text": txt}
                         for lbl, txt in FEW_SHOT_EXAMPLES],
        },
    }
