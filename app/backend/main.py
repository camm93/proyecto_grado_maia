"""
MAIA — Análisis Retórico + Detección de Contribuciones
=======================================================
Composición de la app FastAPI: middleware, rutas, montaje del frontend
y SPA fallback. La lógica vive en `core/` y los endpoints en `api/`.

Endpoints:
  POST /api/segment        → Tarea 1: segmentación retórica (T1)
  POST /api/contributions  → Tarea 2: detección de contribuciones (T2)
  GET  /api/models         → catálogo de modelos T1 y T2
  GET  /api/lexicon        → patrones regex compartidos con el frontend
  GET  /health             → estado del servicio

Arranque:
  cd app
  pip install -r backend/requirements.txt
  uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .api import system, segment, contributions
from .core.catalog import MODELS_T1
from .model_loader import get_t1_classifier, ModelNotAvailableError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("maia")

# ── Localización del frontend ────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()


def _find_frontend() -> Path:
    """Resuelve el directorio del frontend en este orden:
    1. variable de entorno FRONTEND_DIR
    2. ../frontend respecto a este archivo (layout estándar)
    3. búsqueda hacia arriba como fallback
    """
    env = os.getenv("FRONTEND_DIR")
    if env:
        p = Path(env).resolve()
        if (p / "index.html").exists():
            return p

    default = (_HERE.parent / "frontend").resolve()
    if (default / "index.html").exists():
        return default

    for candidate in [_HERE, _HERE.parent, _HERE.parent.parent, _HERE.parent.parent.parent]:
        f = candidate / "frontend"
        if f.is_dir() and (f / "index.html").exists():
            return f
    return default


FRONTEND_DIR = _find_frontend()
log.info(f"Frontend detectado en: {FRONTEND_DIR}")


# ── Lifespan: pre-carga del modelo T1 al startup ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-carga el modelo T1 SciBETO V5 al startup si está disponible y si la
    variable T1_PRELOAD no está en "false". Esto evita que el primer request
    pague la latencia de cargar 1.3 GB de pesos.

    Controlado por:
        T1_PRELOAD       — "false" para deshabilitar (default: true)
        T1_PRELOAD_ID    — modelo a precargar (default: "scibert-es")

    El fallo de pre-carga NO bloquea el startup: el endpoint hará fallback
    a heurística cuando no encuentre los pesos.
    """
    preload_enabled = os.getenv("T1_PRELOAD", "true").lower() not in ("false", "0", "no")
    preload_id = os.getenv("T1_PRELOAD_ID", "scibert-es")

    if preload_enabled:
        info = next((m for m in MODELS_T1 if m["id"] == preload_id), None)
        if info and info["available"]:
            try:
                log.info(f"Pre-cargando modelo T1 '{preload_id}' al startup...")
                classifier = get_t1_classifier(preload_id)
                classifier.warmup()
                log.info(
                    f"✅ Modelo T1 '{preload_id}' pre-cargado en device={classifier.device}. "
                    "El primer request no pagará latencia de carga."
                )
            except ModelNotAvailableError as e:
                log.warning(f"No se pudo pre-cargar T1 '{preload_id}': {e}. "
                            "El endpoint hará fallback a heurística si se solicita.")
            except Exception as e:
                log.exception(f"Error inesperado pre-cargando T1 '{preload_id}': {e}")
        else:
            log.info(f"T1 '{preload_id}' no available; skip pre-carga.")
    else:
        log.info("T1_PRELOAD=false; skip pre-carga del modelo.")

    yield  # ← acá la app sirve requests

    log.info("Shutdown del servidor MAIA.")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MAIA — API de Análisis Retórico",
    description=(
        "**Tarea 1** `POST /api/segment` — Segmentación retórica (8 etiquetas)\n\n"
        "**Tarea 2** `POST /api/contributions` — Detección de contribuciones científicas\n\n"
        "Flujo obligatorio: primero `/api/segment`, luego `/api/contributions` "
        "con los segmentos devueltos por T1."
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.include_router(system.router)
app.include_router(segment.router)
app.include_router(contributions.router)


# ── SPA fallback (con guard para /api/*) ─────────────────────────────────────
@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa(full_path: str = ""):
    # Si una ruta /api/* llega aquí es porque no fue resuelta por ningún router:
    # devolver 404 JSON explícito en vez de servir el HTML del SPA.
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(404, f"Endpoint API no encontrado: /{full_path}")

    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    raise HTTPException(404, "Frontend no encontrado.")
