"""
Catálogo de modelos T1 y T2 + resolución del modelo a usar.

La disponibilidad de modelos basados en API depende de variables de entorno
leídas en import time (fastAPI no recarga el módulo entre requests).

La disponibilidad de modelos locales (encoders fine-tuned + Llama 4-bit)
depende de que los pesos estén descargados en MODELS_DIR o que el modelo HF
sea descargable con el token configurado.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from fastapi import HTTPException

log = logging.getLogger("maia.catalog")


def _api_key_present(*names: str) -> bool:
    return any(bool(os.getenv(n)) for n in names)


def _llama_available() -> bool:
    """Llama está disponible si hay HF_TOKEN + GPU disponible.

    Verificación liviana — el chequeo real (carga del modelo) ocurre en
    runtime via llm_service.get_llama_service() con su propia gestión
    de errores.
    """
    if not os.getenv("HF_TOKEN"):
        return False
    # Si LLAMA_ENABLED=false, deshabilitar explícitamente (útil en CI)
    if os.getenv("LLAMA_ENABLED", "true").lower() == "false":
        return False
    # No probamos CUDA acá para no requerir torch en import time del catálogo;
    # si CUDA falta, llm_service tirará ModelNotAvailableError en runtime.
    return True


def _local_weights_present(model_dir_name: str) -> bool:
    """Verifica si existen pesos REALES del modelo para `MODELS_DIR/<model_dir_name>`.

    Excluye archivos *.bin que son metadata del Trainer (`training_args.bin`,
    `optimizer.bin`, etc.) y no pesos del encoder. Los pesos reales tienen
    nombres canónicos: `pytorch_model.bin`, `model.safetensors`, o variantes
    sharded `model-00001-of-XXXXX.safetensors`.
    """
    models_root = Path(os.getenv("MODELS_DIR", "./models"))
    p = models_root / model_dir_name
    if not p.exists() or not (p / "config.json").exists():
        return False

    weight_files = (
        list(p.glob("pytorch_model.bin"))
        + list(p.glob("pytorch_model-*-of-*.bin"))
        + list(p.glob("model.safetensors"))
        + list(p.glob("model-*-of-*.safetensors"))
    )
    return bool(weight_files)


# ── T1 ──────────────────────────────────────────────────────────────────────
MODELS_T1: list[dict] = [
    {"id": "heuristic", "name": "Heuristico (reglas lexicas)", "family": "heuristic",
     "available": True, "task": "T1",
     "description": "Reglas lexico-posicionales. Siempre disponible como baseline."},
    {"id": "scibeto-es-t1", "name": "SciBETO-ES T1 (SciBETO-large v12 fine-tuned)",
     "family": "encoder",
     "available": _local_weights_present("scibeto-es-t1"), "task": "T1",
     "description": "SciBETO-large fine-tuned (run task1-scibeto-v12-cris) con 7 "
                    "clases retoricas (sin CONTR). F1 macro = 0.40 sobre 1401 "
                    "samples gold humano (200 por clase, anotacion de 2 humanos). "
                    "Re-entrenado en v6 sobre data batches 1-4. Preprocesamiento "
                    "dinamico con regex extractor + mascara de matched_keywords."},
    {"id": "llama-3.1-8b-t1", "name": "Llama 3.1 8B (T1, few-shot k=14)",
     "family": "llm_open",
     "available": _llama_available(), "task": "T1",
     "description": "Llama 3.1 8B-Instruct en 4-bit, few-shot con 14 ejemplos "
                    "(2 por clase x 7 clases sin CONTR). F1 macro = 0.322 sobre "
                    "1400 gold humano. Parser tolerante con _LLM_LABEL_MAP. "
                    "Requiere HF_TOKEN + GPU local con >=10 GB VRAM."},
    {"id": "llama-3.1-8b-t1-zs", "name": "Llama 3.1 8B (T1, zero-shot)",
     "family": "llm_open",
     "available": _llama_available(), "task": "T1",
     "description": "Llama 3.1 8B-Instruct en 4-bit, zero-shot (sin ejemplos). "
                    "F1 macro = 0.234 sobre 1400 gold humano. Variante de "
                    "ablacion agregada en v3 para contrastar el efecto del "
                    "few-shot: en este modelo el FS mejora +8.85 pp vs ZS. "
                    "Requiere HF_TOKEN + GPU local con >=10 GB VRAM."},
    {"id": "gpt-4o-mini-t1", "name": "GPT-4o-mini (T1, few-shot k=14 + JSON mode)",
     "family": "llm_api",
     "available": _api_key_present("OPENAI_API_KEY"), "task": "T1",
     "description": "GPT-4o-mini via OpenAI API, few-shot con 14 ejemplos + JSON "
                    "mode + retry exponencial + concurrencia=8. F1 macro = 0.414 "
                    "sobre 1400 gold humano. Costo aprox. $0.24 USD por 1400 "
                    "fragmentos. Requiere OPENAI_API_KEY."},
    {"id": "gpt-4o-mini-t1-zs", "name": "GPT-4o-mini (T1, zero-shot + JSON mode) -- mejor T1",
     "family": "llm_api",
     "available": _api_key_present("OPENAI_API_KEY"), "task": "T1",
     "description": "GPT-4o-mini via OpenAI API, zero-shot + JSON mode + "
                    "concurrencia=8. F1 macro = 0.447 sobre 1400 gold humano "
                    "(MEJOR de T1; supera al FS-14 que cae a 0.414). Tambien "
                    "mas rapido (2.9 vs 4.3 min) y mas barato ($0.14 vs $0.24 "
                    "USD por 1400 fragmentos). Requiere OPENAI_API_KEY."},
]


# ── T2 ──────────────────────────────────────────────────────────────────────
MODELS_T2: list[dict] = [
    {"id": "heuristic", "name": "Heuristico (reglas lexicas)", "family": "heuristic",
     "available": True, "task": "T2",
     "description": "Patrones explicitos de contribucion. Siempre disponible como baseline."},

    # ── SciBETO encoders (A5 del PDF) ────────────────────────────────────────
    {"id": "scibeto-es-t2", "name": "SciBETO-large text-only (T2) -- recomendado",
     "family": "encoder",
     "available": _local_weights_present("scibeto-es-t2"), "task": "T2",
     "description": "SciBETO-large fine-tuned con texto crudo del fragmento (sin "
                    "metadatos retoricos). F1 macro = 0.852 [IC 95%: 0.793, 0.900] "
                    "sobre 170 gold (kappa Cohen = 0.6995). Threshold optimo = 0.525 "
                    "(calibrable via T2_THRESHOLD)."},
    {"id": "scibeto-es-t2-ctx", "name": "SciBETO-large + contexto retorico (T2)",
     "family": "encoder",
     "available": _local_weights_present("scibeto-es-t2-ctx"), "task": "T2",
     "description": "Variante con prefijo '[SECCION:X] [POS:Y]'. F1 macro = 0.798 "
                    "[IC 95%: 0.736, 0.858]. NO superior al baseline text-only: "
                    "el ablation revela atajos {t1_label,pos} -> clase. Disponible "
                    "como ablacion del A5 del PDF; en produccion se recomienda el "
                    "baseline."},

    # ── LLMs open-weight (A6 del PDF) ────────────────────────────────────────
    {"id": "llama-3.1-8b-t2", "name": "Llama 3.1 8B (T2, zero-shot)",
     "family": "llm_open",
     "available": _llama_available(), "task": "T2",
     "description": "Llama 3.1 8B-Instruct en 4-bit, zero-shot binario. F1 macro "
                    "= 0.393 sobre 170 gold (colapsa hacia clase negativa). "
                    "Requiere HF_TOKEN + GPU local con >=10 GB VRAM."},
    {"id": "llama-3.1-8b-t2-fs8", "name": "Llama 3.1 8B (T2, few-shot k=8)",
     "family": "llm_open",
     "available": _llama_available(), "task": "T2",
     "description": "Llama 3.1 8B-Instruct con 8 ejemplos few-shot curados "
                    "(4 positivos + 4 negativos balanceados, orden alternado). "
                    "Mejor que zero-shot pero aun inferior a SciBETO."},

    # ── LLMs comerciales (A7 del PDF) ────────────────────────────────────────
    {"id": "gpt-4o-mini-t2", "name": "GPT-4o-mini (T2, zero-shot)",
     "family": "llm_api",
     "available": _api_key_present("OPENAI_API_KEY"), "task": "T2",
     "description": "GPT-4o-mini via OpenAI API, zero-shot binario. F1 macro = "
                    "0.423 sobre 170 gold. Costo aprox. $0.001 por fragmento. "
                    "Requiere OPENAI_API_KEY."},
    {"id": "gpt-4o-mini-t2-fs8", "name": "GPT-4o-mini (T2, few-shot k=8)",
     "family": "llm_api",
     "available": _api_key_present("OPENAI_API_KEY"), "task": "T2",
     "description": "GPT-4o-mini con 8 ejemplos few-shot curados (4 pos + 4 neg, "
                    "orden alternado). Mejor desempeno que zero-shot a costo "
                    "aprox. 2x mas por consumo de tokens."},
]


def refresh_catalog() -> None:
    """Recompute el flag `available` de cada modelo basado en el estado ACTUAL
    de las env vars y los pesos en disco.

    Necesario porque `MODELS_T1` y `MODELS_T2` se construyen al import time
    del módulo. Si el usuario edita `.env` o descarga pesos después de
    arrancar uvicorn, los valores quedan stale. Llamar a refresh_catalog()
    antes de servir `/api/models` o de hacer resolve_model() garantiza que
    los valores reflejen el estado actual.

    Es barato: solo lee env vars y hace stat() en MODELS_DIR. No carga modelos.
    """
    for m in MODELS_T1 + MODELS_T2:
        family = m["family"]
        if family == "heuristic":
            # heurística siempre disponible
            m["available"] = True
        elif family == "encoder":
            m["available"] = _local_weights_present(m["id"])
        elif family == "llm_open":
            m["available"] = _llama_available()
        elif family == "llm_api":
            m["available"] = _api_key_present("OPENAI_API_KEY")


def resolve_model(model_id: str, catalog: list[dict]) -> tuple[dict, str, str]:
    """Devuelve (info, mode, display_name) para un model_id contra el catálogo dado.

    Recomputa `available` antes de resolver (refresh_catalog), así que las
    keys/pesos cambiados en runtime se reflejan sin reiniciar uvicorn.

    Si el modelo no está disponible aún, hace fallback a heurístico y lo deja
    explícito en el nombre devuelto.
    """
    refresh_catalog()

    info = next((m for m in catalog if m["id"] == model_id), None)
    if not info:
        raise HTTPException(404, f"Modelo '{model_id}' no encontrado.")

    is_heuristic = model_id == "heuristic" or info["family"] == "heuristic"
    if not info["available"] and not is_heuristic:
        log.warning(f"Modelo {model_id} no disponible -- usando heuristico como fallback.")

    mode = "heuristic" if (is_heuristic or not info["available"]) else info["family"]
    name = info["name"] + ("" if info["available"] else " [fallback heuristico]")
    return info, mode, name
