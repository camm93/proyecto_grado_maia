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

# ── Cargar .env ANTES de importar catalog/llm_service ────────────────────────
# Si .env existe en cwd o en el dir del proyecto, sus variables se inyectan
# en os.environ. Esto es crítico porque catalog._llama_available() y otros
# helpers chequean OPENAI_API_KEY / HF_TOKEN al import time del módulo.
# Si .env se carga DESPUÉS, los flags quedan en False aunque las keys estén
# en el archivo.
try:
    from dotenv import load_dotenv
    # Buscar .env en cwd primero, luego en parent del backend/
    _project_root = Path(__file__).resolve().parent.parent
    for _candidate in (Path.cwd() / ".env", _project_root / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            logging.getLogger("maia").info(f"Cargado .env desde {_candidate}")
            break
except ImportError:
    # python-dotenv no instalado; las env vars deben venir del shell.
    # uvicorn[standard] lo incluye por default; si no, instalar con:
    #   pip install python-dotenv
    logging.getLogger("maia").warning(
        "python-dotenv no instalado; .env NO se cargara automaticamente. "
        "Setear env vars en el shell o instalar: pip install python-dotenv"
    )

# ── HF_HOME: cache de HuggingFace dentro de models/ ─────────────────────────
# Por default huggingface_hub guarda los modelos en ~/.cache/huggingface, que
# en muchos VPS (incluida la EC2 g4dn.xlarge con AMI default) cae en el disco
# root de 8-30 GB. Llama 3.1 8B ocupa ~16 GB y revienta ese disco.
#
# Solución: apuntar el caché HF a <project_root>/models/.hf_cache/ — el mismo
# volumen donde viven los SciBETO (que el usuario ya proveyó con espacio
# suficiente). Esto garantiza que TODO lo descargable (SciBETOs gestionados
# por model_sync + Llama gestionado por HF) viva en un único volumen.
#
# La ruta usa pathlib.Path para que funcione igual en Linux/macOS/Windows
# (los separadores se resuelven automáticamente). El prefijo `.hf_cache`
# (oculto en Linux) deja claro que es interno y evita que model_sync.py
# lo confunda con un modelo SciBETO real.
#
# Override: si HF_HOME está seteado en .env o env vars del shell, respetamos
# eso (caso: EBS adicional montado en otro path).
if not os.getenv("HF_HOME"):
    _project_root = Path(__file__).resolve().parent.parent
    _default_hf_home = _project_root / "models" / ".hf_cache"
    _default_hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_default_hf_home)
    logging.getLogger("maia").info(f"HF_HOME no seteado; usando {_default_hf_home}")
else:
    logging.getLogger("maia").info(f"HF_HOME (custom): {os.environ['HF_HOME']}")

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .api import system, segment, contributions, extract
from .core.catalog import MODELS_T1, MODELS_T2
from .model_loader import get_t1_classifier, get_t2_classifier, ModelNotAvailableError

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


# ── Lifespan: pre-carga de modelos al startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-carga modelos al startup para evitar latencia en el primer request.

    Por default se precargan:
        - T1 SciBETO-ES (scibeto-es-t1)
        - T2 SciBETO-ES baseline (scibeto-es-t2)
        - T2 SciBETO-ES ctx (scibeto-es-t2-ctx)
        - Llama 3.1 8B (Llama es lazy por default — toma 10-20 min en CPU
          y 1-2 min en GPU descargar + cargar 16 GB de pesos).

    Variables de control:
        T1_PRELOAD          — "false" para deshabilitar (default: true)
        T1_PRELOAD_ID       — modelo T1 (default: "scibeto-es-t1")
        T2_PRELOAD          — "false" para deshabilitar (default: true)
        T2_PRELOAD_ID       — modelo T2 baseline (default: "scibeto-es-t2")
        T2_CTX_PRELOAD      — "false" para deshabilitar (default: true)
        T2_CTX_PRELOAD_ID   — modelo T2 ctx (default: "scibeto-es-t2-ctx")
        LLAMA_PRELOAD       — "false" para deshabilitar Llama (default: true).
                              ¡OJO! En CPU agrega 10-20 min al startup (descarga
                              de 16 GB desde HF + load). En GPU agrega 1-2 min.
                              Dejar true en producción; en dev local con poca
                              RAM (<32 GB) o sin GPU, conviene desactivar.

    El fallo de pre-carga NO bloquea el startup: cada endpoint hace fallback
    a heurística cuando no encuentre los pesos / credenciales.

    Estimado de uso de recursos con TODOS los preloads activos:
        GPU (T4 16 GB):   ~13 GB VRAM (3× SciBETO FP16 + Llama 4-bit)
        CPU only:         ~22 GB RAM (3× SciBETO FP32 + Llama FP16)
    """
    # ── T1 SciBETO ────────────────────────────────────────────────────────────
    preload_t1 = os.getenv("T1_PRELOAD", "true").lower() not in ("false", "0", "no")
    preload_t1_id = os.getenv("T1_PRELOAD_ID", "scibeto-es-t1")

    if preload_t1:
        info = next((m for m in MODELS_T1 if m["id"] == preload_t1_id), None)
        if info and info["available"]:
            try:
                log.info(f"Pre-cargando modelo T1 '{preload_t1_id}' al startup...")
                classifier = get_t1_classifier(preload_t1_id)
                classifier.warmup()
                log.info(
                    f"[OK] Modelo T1 '{preload_t1_id}' pre-cargado en device={classifier.device}. "
                    "El primer request no pagara latencia de carga."
                )
            except ModelNotAvailableError as e:
                log.warning(f"No se pudo pre-cargar T1 '{preload_t1_id}': {e}. "
                            "El endpoint hara fallback a heuristica si se solicita.")
            except Exception as e:
                log.exception(f"Error inesperado pre-cargando T1 '{preload_t1_id}': {e}")
        else:
            log.info(f"T1 '{preload_t1_id}' no available; skip pre-carga.")
    else:
        log.info("T1_PRELOAD=false; skip pre-carga del modelo T1.")

    # ── T2 SciBETO baseline ───────────────────────────────────────────────────
    preload_t2 = os.getenv("T2_PRELOAD", "true").lower() not in ("false", "0", "no")
    preload_t2_id = os.getenv("T2_PRELOAD_ID", "scibeto-es-t2")

    if preload_t2:
        info = next((m for m in MODELS_T2 if m["id"] == preload_t2_id), None)
        if info and info["available"]:
            try:
                log.info(f"Pre-cargando modelo T2 '{preload_t2_id}' al startup...")
                classifier = get_t2_classifier(preload_t2_id)
                classifier.warmup()
                log.info(
                    f"[OK] Modelo T2 '{preload_t2_id}' pre-cargado en device={classifier.device}. "
                    "El primer request no pagara latencia de carga."
                )
            except ModelNotAvailableError as e:
                log.warning(f"No se pudo pre-cargar T2 '{preload_t2_id}': {e}. "
                            "El endpoint hara fallback a heuristica si se solicita.")
            except Exception as e:
                log.exception(f"Error inesperado pre-cargando T2 '{preload_t2_id}': {e}")
        else:
            log.info(f"T2 '{preload_t2_id}' no available; skip pre-carga.")
    else:
        log.info("T2_PRELOAD=false; skip pre-carga del modelo T2.")

    # ── T2 SciBETO ctx (ablation) ─────────────────────────────────────────────
    preload_t2_ctx = os.getenv("T2_CTX_PRELOAD", "true").lower() not in ("false", "0", "no")
    preload_t2_ctx_id = os.getenv("T2_CTX_PRELOAD_ID", "scibeto-es-t2-ctx")

    if preload_t2_ctx:
        info = next((m for m in MODELS_T2 if m["id"] == preload_t2_ctx_id), None)
        if info and info["available"]:
            try:
                log.info(f"Pre-cargando modelo T2 ctx '{preload_t2_ctx_id}' al startup...")
                classifier = get_t2_classifier(preload_t2_ctx_id)
                classifier.warmup()
                log.info(
                    f"[OK] Modelo T2 ctx '{preload_t2_ctx_id}' pre-cargado en device={classifier.device}."
                )
            except ModelNotAvailableError as e:
                log.warning(f"No se pudo pre-cargar T2 ctx '{preload_t2_ctx_id}': {e}.")
            except Exception as e:
                log.exception(f"Error inesperado pre-cargando T2 ctx '{preload_t2_ctx_id}': {e}")
        else:
            log.info(f"T2 ctx '{preload_t2_ctx_id}' no available; skip pre-carga.")
    else:
        log.info("T2_CTX_PRELOAD=false; skip pre-carga del modelo T2 ctx.")

    # ── Llama 3.1 8B (preload por default — toma tiempo y RAM, pero evita
    # que el usuario pague esa latencia en el primer request) ────────────────
    # En GPU (T4 g4dn.xlarge): 1-2 min de descarga + carga
    # En CPU (dev local Windows): 10-20 min de descarga (16 GB) + load
    # Para desactivar (típicamente en laptop con <32 GB RAM): LLAMA_PRELOAD=false
    preload_llama = os.getenv("LLAMA_PRELOAD", "true").lower() not in ("false", "0", "no")

    if preload_llama:
        # Verificar si Llama tiene credenciales antes de intentar cargarlo,
        # para evitar logs ruidosos cuando el usuario no tiene HF_TOKEN.
        if not os.getenv("HF_TOKEN"):
            log.warning("LLAMA_PRELOAD=true pero HF_TOKEN no configurado; skip.")
        else:
            try:
                log.info(
                    "Pre-cargando Llama 3.1 8B al startup... "
                    "(en GPU: 1-2 min; en CPU: 10-20 min para descargar 16 GB + load)"
                )
                from .core.llm_service import get_llama_service
                llama = get_llama_service()

                # Warmup inference SOLO en GPU. En CPU, una inferencia con el
                # prompt T1 completo tarda 10-25 min (1000 tokens de input,
                # ~0.5s por token). El warmup sirve para inicializar CUDA
                # kernels — en CPU no hace nada útil, solo bloquea el startup.
                if llama.device == "cuda":
                    _label, _raw, _elapsed = llama.predict_t1(
                        "Texto de warmup para inicializar kernels."
                    )
                    log.info(
                        f"[OK] Llama 3.1 8B pre-cargado en device={llama.device}. "
                        f"Warmup inference: {_elapsed*1000:.0f}ms. "
                        "Los requests a llama-3.1-8b-t* no pagaran latencia de carga."
                    )
                else:
                    log.info(
                        f"[OK] Llama 3.1 8B pre-cargado en device={llama.device}. "
                        "Warmup skipped en CPU (tomaria 10-25 min sin beneficio). "
                        "El primer request real pagara esa latencia de inferencia."
                    )
            except ModelNotAvailableError as e:
                log.warning(f"No se pudo pre-cargar Llama: {e}. "
                            "Los endpoints LLM haran fallback a heuristica.")
            except MemoryError as e:
                log.error(
                    f"OOM pre-cargando Llama: {e}. La maquina no tiene "
                    "suficiente RAM/VRAM. Setear LLAMA_PRELOAD=false o usar "
                    "una maquina con mas memoria. Los endpoints LLM haran fallback."
                )
            except Exception as e:
                log.exception(f"Error inesperado pre-cargando Llama: {e}")
    else:
        log.info("LLAMA_PRELOAD=false; Llama se cargara lazy en el primer request "
                 "(esto agregara 1-20 min de latencia al primer uso segun hardware).")

    yield  # ← acá la app sirve requests

    log.info("Shutdown del servidor MAIA.")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MAIA -- API de Analisis Retorico",
    description=(
        "**Tarea 1** `POST /api/segment` -- Segmentacion retorica (8 etiquetas)\n\n"
        "**Tarea 2** `POST /api/contributions` -- Deteccion de contribuciones cientificas\n\n"
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

# Rate limiting por IP (cap 1000 req/hora por default; configurable via
# RATE_LIMIT_PER_HOUR). Aplica solo a /api/*. Whitelist via RATE_LIMIT_WHITELIST
# (CSV). Para deshabilitar: setear RATE_LIMIT_PER_HOUR=0 (no implementado todavia
# explicitamente; en su lugar usar un numero muy alto como 999999).
from .middleware.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.include_router(system.router)
app.include_router(segment.router)
app.include_router(contributions.router)
app.include_router(extract.router)


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
