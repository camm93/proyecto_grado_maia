"""
backend/utils/model_sync.py -- Sincronización de pesos desde Google Drive.
========================================================================

Verificación dual de integridad:

  1. Al descargar un ZIP (cuando falta algún archivo extraído o se detecta
     corrupción): bajar con `gdown >= 5.0` usando `fuzzy=True` para manejar
     la pantalla intermedia de virus-scan de Drive para archivos >100MB,
     verificar SHA256 del ZIP completo contra `manifest.models[id].zip.sha256`,
     re-descargar hasta 3 veces con backoff exponencial ante mismatch.

  2. Al arrancar el backend (cada uvicorn startup, antes de servir requests):
     para cada modelo, verificar SHA256 de cada archivo individual
     `required: true` contra `manifest.models[id].files[].sha256`. Si falta
     archivo o hash mismatch -> trigger del paso 1.

Convenciones:
  - El manifest es la fuente única de verdad. Los nombres de directorios
    destino vienen de `manifest.models[id].extracted_dir`.
  - Los archivos `required: false` (ej. `training_args.bin`) se ignoran en
    la verificación: ni se chequea su existencia ni su hash. Si están en
    el ZIP se descomprimen, pero su ausencia/corrupción no triggerea
    re-descarga.
  - Los modelos en `manifest.llms` NO requieren descarga (Llama desde HF
    Hub, GPT vía API); solo se iteran los de `manifest.models`.

Variables de entorno:
  MODELS_DIR              destino de los pesos (default: /app/models)
  MANIFEST_PATH           path al release manifest (default:
                          /app/release_manifest.json)
  DOWNLOAD_FAIL_POLICY    abort | warn (default: abort)
                            abort -> sys.exit(1) ante fallo definitivo
                            warn  -> log ERROR y continuar (modelo cae a
                                    fallback heurístico)
  MIN_FREE_DISK_GB        chequeo mínimo de espacio antes de bajar
                          (default: 10)

CLI:
  python -m backend.utils.model_sync ensure_all
  python -m backend.utils.model_sync ensure <model_id>
  python -m backend.utils.model_sync verify           # sin descarga

Diseño para tests:
  - `load_manifest(path)` es pura y no usa cache.
  - `get_manifest()` cachea para runtime; los tests pueden resetear con
    `reset_caches()` después de monkeypatchear `MANIFEST_PATH`.
  - `gdown` se importa lazily dentro de `download_zip_from_drive()`, así
    los tests sin gdown instalado pueden monkeypatchear esa función.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

log = logging.getLogger("maia.model_sync")

# -- Defaults / env -----------------------------------------------------------
DEFAULT_MANIFEST_PATH = "/app/release_manifest.json"
DEFAULT_MODELS_DIR = "/app/models"
DEFAULT_MIN_FREE_DISK_GB = 8
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 5

_MANIFEST_CACHE: dict | None = None
_MODELS_DIR_CACHE: Path | None = None


class ModelSyncError(RuntimeError):
    """Error definitivo durante sync (post-reintentos o config inválida)."""


# -- Manifest y paths ---------------------------------------------------------
def load_manifest(path: Path | None = None) -> dict:
    """Lee y parsea el release manifest.

    Sin cache -- usa esto en tests; runtime debe usar `get_manifest()`.

    Resolución del path:
      1. Argumento explícito `path`
      2. Variable de entorno MANIFEST_PATH
      3. Default /app/release_manifest.json
      4. Fallback ./release_manifest.json (dev local)
    """
    if path is None:
        env = os.getenv("MANIFEST_PATH")
        path = Path(env) if env else Path(DEFAULT_MANIFEST_PATH)

    if not path.exists():
        # Fallback dev local (cwd)
        local = Path("release_manifest.json")
        if local.exists():
            path = local
        else:
            raise ModelSyncError(
                f"No se encontro el manifest en {path}. Define MANIFEST_PATH "
                "o ubica el archivo en /app/ o en el cwd."
            )

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_manifest() -> dict:
    """Manifest cacheado para runtime. Reset con `reset_caches()`."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        _MANIFEST_CACHE = load_manifest()
    return _MANIFEST_CACHE


def get_models_dir() -> Path:
    """Directorio destino de pesos. Cacheado. Reset con `reset_caches()`."""
    global _MODELS_DIR_CACHE
    if _MODELS_DIR_CACHE is None:
        _MODELS_DIR_CACHE = Path(os.getenv("MODELS_DIR", DEFAULT_MODELS_DIR))
    return _MODELS_DIR_CACHE


def reset_caches() -> None:
    """Limpia caches de manifest y MODELS_DIR. Para uso en tests."""
    global _MANIFEST_CACHE, _MODELS_DIR_CACHE
    _MANIFEST_CACHE = None
    _MODELS_DIR_CACHE = None


def _get_model_entry(model_id: str) -> dict:
    """Devuelve `manifest.models[model_id]` o levanta ModelSyncError."""
    manifest = get_manifest()
    models = manifest.get("models", {})
    if model_id not in models:
        raise ModelSyncError(
            f"Model_id '{model_id}' no esta en manifest.models. "
            f"Disponibles: {sorted(models.keys())}"
        )
    return models[model_id]


# -- Hashing ------------------------------------------------------------------
def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Calcula SHA256 streaming. chunk=8MB; balance entre RAM y throughput."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# -- Locking (multi-worker safe) ----------------------------------------------
@contextmanager
def _sync_lock(models_dir: Path) -> Iterator[None]:
    """Lock file para evitar que dos workers de uvicorn bajen los mismos
    pesos en paralelo.

    Implementación con `fcntl.flock` (Linux/macOS only). En Windows hace
    no-op con warning -- el deploy AWS es Linux así que no afecta
    producción. El lock se libera automáticamente al salir del with o
    si el proceso muere.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    lock_path = models_dir / ".sync.lock"
    try:
        import fcntl
    except ImportError:
        log.warning("fcntl no disponible (Windows); sync sin lock entre procesos.")
        yield
        return

    f = lock_path.open("a+")
    try:
        log.info(f"Adquiriendo lock {lock_path}...")
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        log.info("Lock adquirido.")
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


# -- Pre-check de espacio -----------------------------------------------------
def _check_disk_space(models_dir: Path, manifest: dict,
                      active_only: bool = False) -> None:
    """Aborta temprano si no hay suficiente espacio para descargar +
    descomprimir todos los modelos del manifest.

    Heurística: 2x la suma de `size_mb` (ZIP + extracted) + 1 GB de margen,
    floor en MIN_FREE_DISK_GB.

    Si `active_only=True`, modelos con `deprecated: true` se omiten del
    cálculo (uso desde `ensure_all_models`).
    """
    models = manifest.get("models", {})
    if active_only:
        models = {k: v for k, v in models.items()
                  if not v.get("deprecated", False)}
    needed_mb = sum(
        2 * float(m.get("zip", {}).get("size_mb", 0))
        for m in models.values()
    )
    needed_mb += 1024  # margen 1 GB

    min_gb = float(os.getenv("MIN_FREE_DISK_GB", DEFAULT_MIN_FREE_DISK_GB))
    needed_mb = max(needed_mb, min_gb * 1024)

    models_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(models_dir).free
    free_mb = free_bytes / (1024 * 1024)

    log.info(f"Espacio libre en {models_dir}: {free_mb:.0f} MB "
             f"(requerido: {needed_mb:.0f} MB)")

    if free_mb < needed_mb:
        raise ModelSyncError(
            f"Espacio insuficiente en {models_dir}: "
            f"{free_mb:.0f} MB libres, {needed_mb:.0f} MB requeridos. "
            "Libera espacio o redirige MODELS_DIR a un volumen mayor."
        )


# -- Fail policy --------------------------------------------------------------
def _handle_fail(model_id: str, msg: str) -> None:
    """Aplica DOWNLOAD_FAIL_POLICY.

    abort (default) -> ModelSyncError (que el CLI traduce a sys.exit(1))
    warn            -> log ERROR y retornar; el modelo quedará no-disponible
                      y el backend caerá a fallback heurístico
    """
    policy = os.getenv("DOWNLOAD_FAIL_POLICY", "abort").lower().strip()
    full = f"[{model_id}] {msg}"
    if policy == "warn":
        log.error(f"{full} (DOWNLOAD_FAIL_POLICY=warn; backend continuara "
                  "con fallback heuristico para este modelo)")
        return
    raise ModelSyncError(full)


# -- Verificación: ZIP en cache -----------------------------------------------
def verify_zip(model_id: str) -> bool:
    """Verifica que el ZIP en `{MODELS_DIR}/.cache/{filename}` exista y
    tenga el SHA256 declarado en el manifest. True si OK."""
    entry = _get_model_entry(model_id)
    zip_meta = entry["zip"]
    cache_path = get_models_dir() / ".cache" / zip_meta["filename"]

    if not cache_path.exists():
        log.info(f"[{model_id}] ZIP no presente en cache ({cache_path}).")
        return False

    log.info(f"[{model_id}] Verificando SHA256 del ZIP {cache_path}...")
    actual = _sha256_file(cache_path)
    expected = zip_meta["sha256"]

    if actual != expected:
        log.warning(f"[{model_id}] SHA mismatch del ZIP:\n"
                    f"  esperado: {expected}\n  obtenido: {actual}")
        return False

    log.info(f"[{model_id}] ZIP OK (sha256 coincide).")
    return True


# -- Verificación: archivos extraídos -----------------------------------------
def verify_extracted_files(model_id: str) -> tuple[bool, list[str]]:
    """Verifica cada archivo `required: true` del modelo extraído.

    Devuelve (todo_ok, lista_de_rutas_problemáticas). Una ruta entra en la
    lista si:
      - el archivo no existe, o
      - su SHA256 no coincide con el manifest.

    Los archivos `required: false` se ignoran completamente (política
    lenient). Tampoco se reporta si un archivo extra no listado está
    presente.
    """
    entry = _get_model_entry(model_id)
    extracted_dir = get_models_dir() / entry["extracted_dir"]
    issues: list[str] = []

    for finfo in entry.get("files", []):
        if not finfo.get("required", True):
            continue

        fpath = extracted_dir / finfo["path"]
        if not fpath.exists():
            log.info(f"[{model_id}] Falta archivo {fpath} (required).")
            issues.append(str(fpath))
            continue

        actual = _sha256_file(fpath)
        if actual != finfo["sha256"]:
            log.warning(f"[{model_id}] SHA mismatch en {finfo['path']}:\n"
                        f"  esperado: {finfo['sha256']}\n  obtenido: {actual}")
            issues.append(str(fpath))

    return (len(issues) == 0, issues)


# -- Descarga desde Drive -----------------------------------------------------
def download_zip_from_drive(model_id: str) -> Path:
    """Descarga el ZIP desde Google Drive con `gdown` (fuzzy=True maneja
    la pantalla intermedia de virus-scan para archivos >100MB).

    Reintenta hasta DEFAULT_MAX_RETRIES con backoff exponencial. Después
    de cada descarga verifica SHA256 del ZIP. Si el SHA final tras 3
    intentos sigue sin coincidir, levanta ModelSyncError.

    Devuelve el Path absoluto al ZIP descargado y verificado.
    """
    entry = _get_model_entry(model_id)
    zip_meta = entry["zip"]
    drive_url = zip_meta["drive_url"]
    filename = zip_meta["filename"]

    cache_dir = get_models_dir() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / filename

    # Lazy import: tests pueden monkeypatchear `gdown.download`
    try:
        import gdown
    except ImportError as e:
        raise ModelSyncError(
            f"gdown no instalado. Agregalo a requirements.txt: gdown>=6.0.0 "
            f"(error original: {e})"
        )

    # Compat gdown 5.x <-> 6.x:
    # - gdown <6.0 requiere fuzzy=True para URLs de share-link
    #   (https://drive.google.com/file/d/.../view).
    # - gdown >=6.0 eliminó fuzzy (#455) y extrae el file_id de cualquier
    #   formato de URL automáticamente.
    # Detectamos la firma para mantener compat en ambas versiones.
    import inspect
    _gdown_kwargs_extra: dict = {}
    try:
        _params = inspect.signature(gdown.download).parameters
        if "fuzzy" in _params:
            _gdown_kwargs_extra["fuzzy"] = True
    except (TypeError, ValueError):
        # Si inspect no puede leer la firma (raro), seguimos sin fuzzy.
        pass

    last_err: str = ""
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        log.info(f"[{model_id}] Descarga intento {attempt}/{DEFAULT_MAX_RETRIES} "
                 f"desde {drive_url} -> {out_path}")
        # Si quedo un archivo parcial de un intento previo, eliminarlo
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass

        try:
            result = gdown.download(
                url=drive_url, output=str(out_path), quiet=False,
                **_gdown_kwargs_extra,
            )
            if result is None or not out_path.exists():
                last_err = "gdown.download devolvio None o el archivo no quedo en disco"
                log.warning(f"[{model_id}] Descarga fallida: {last_err}")
            elif verify_zip(model_id):
                log.info(f"[{model_id}] Descarga OK en intento {attempt}.")
                return out_path
            else:
                last_err = "SHA256 del ZIP descargado no coincide con manifest"
                log.warning(f"[{model_id}] {last_err}")
        except Exception as e:  # gdown levanta varios tipos segun el error
            last_err = f"{type(e).__name__}: {e}"
            log.warning(f"[{model_id}] Excepcion de gdown: {last_err}")

        if attempt < DEFAULT_MAX_RETRIES:
            wait = DEFAULT_BACKOFF_BASE_S * (2 ** (attempt - 1))
            log.info(f"[{model_id}] Backoff {wait}s antes del siguiente intento.")
            time.sleep(wait)

    raise ModelSyncError(
        f"Descarga fallida tras {DEFAULT_MAX_RETRIES} intentos. "
        f"Ultimo error: {last_err}"
    )


# -- Descompresión ------------------------------------------------------------
def extract_zip(model_id: str, zip_path: Path) -> None:
    """Descomprime `zip_path` al destino correcto, sobrescribiendo.

    Detecta automáticamente si el ZIP tiene un directorio top-level con
    el mismo nombre que `extracted_dir` (estructura `extracted_dir/...`)
    o si está plano. En ambos casos los archivos finales aterrizan en
    `{MODELS_DIR}/{extracted_dir}/`.
    """
    entry = _get_model_entry(model_id)
    extracted_dir_name = entry["extracted_dir"]
    models_dir = get_models_dir()
    target_dir = models_dir / extracted_dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[{model_id}] Descomprimiendo {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        # ¿El ZIP envuelve todo en un dir top-level == extracted_dir?
        top_dirs = {n.split("/", 1)[0] for n in names if "/" in n}
        flat_files = [n for n in names if "/" not in n]
        wrapped = (
            not flat_files
            and len(top_dirs) == 1
            and next(iter(top_dirs)) == extracted_dir_name
        )
        if wrapped:
            # El ZIP ya trae el dir nivel arriba; descomprimir en models_dir
            zf.extractall(models_dir)
        else:
            # ZIP plano o con prefijo distinto: descomprimir en target_dir
            zf.extractall(target_dir)

    log.info(f"[{model_id}] Descompresion completada.")


# -- Pipeline por modelo ------------------------------------------------------
def ensure_model(model_id: str) -> None:
    """Asegura que el modelo `model_id` tenga sus archivos `required: true`
    descomprimidos y verificados.

    Pipeline:
      1. verify_extracted_files()
         - todo OK -> return (no-op idempotente)
      2. verificar ZIP en cache
         - existe y SHA OK -> extract -> re-verify -> done
         - no existe o SHA fail -> descargar -> extract -> re-verify -> done
      3. Si después de descarga+extract sigue sin verificar ->
         _handle_fail() (abort o warn según policy).
    """
    log.info(f"=== Sync de {model_id} ===")
    entry = _get_model_entry(model_id)
    log.info(f"[{model_id}] extracted_dir={entry['extracted_dir']}, "
             f"size_mb={entry['zip'].get('size_mb', '?')}")

    ok, issues = verify_extracted_files(model_id)
    if ok:
        log.info(f"[{model_id}] [OK] Ya esta descargado y verificado.")
        return

    log.info(f"[{model_id}] Faltan/corruptos: {len(issues)} archivo(s). "
             "Trigger de descarga/extraccion.")

    # Intento 1: usar el ZIP del cache si existe y es válido
    cache_path = get_models_dir() / ".cache" / entry["zip"]["filename"]
    used_cache = False
    if cache_path.exists() and verify_zip(model_id):
        log.info(f"[{model_id}] ZIP en cache valido, descomprimiendo sin re-descarga.")
        try:
            extract_zip(model_id, cache_path)
            used_cache = True
        except (zipfile.BadZipFile, OSError) as e:
            log.warning(f"[{model_id}] Error descomprimiendo cache "
                        f"({e}); forzando re-descarga.")

    if not used_cache:
        try:
            zip_path = download_zip_from_drive(model_id)
            extract_zip(model_id, zip_path)
        except ModelSyncError:
            raise
        except (zipfile.BadZipFile, OSError) as e:
            _handle_fail(model_id, f"Error de descompresion: {e}")
            return

    ok, issues = verify_extracted_files(model_id)
    if not ok:
        _handle_fail(model_id, f"Verificacion final fallo -- archivos: {issues}")
        return

    log.info(f"[{model_id}] [OK] Sync completo y verificado.")


# -- Pipeline global ----------------------------------------------------------
def ensure_all_models() -> None:
    """Sincroniza todos los modelos del manifest secuencialmente.

    Adquiere lock para evitar que múltiples workers compitan. Chequea
    espacio en disco antes de empezar. Itera `manifest.models` (NO
    `manifest.llms`: Llama se baja de HF Hub al primer uso, GPT vía API).

    Modelos marcados con `deprecated: true` se SALTAN en este flujo masivo.
    Si querés bajar un modelo deprecated explícitamente, usá
    `ensure_model(model_id)` directamente.
    """
    manifest = get_manifest()
    models_dir = get_models_dir()
    all_ids = list(manifest.get("models", {}).keys())
    active_ids = [mid for mid in all_ids
                  if not manifest["models"][mid].get("deprecated", False)]
    skipped = [mid for mid in all_ids if mid not in active_ids]

    log.info(f"=== ensure_all_models: {len(active_ids)}/{len(all_ids)} modelo(s) "
             f"activos en manifest -> {models_dir} ===")
    if skipped:
        log.info(f"  Saltando {len(skipped)} modelo(s) deprecated: {skipped}")

    with _sync_lock(models_dir):
        _check_disk_space(models_dir, manifest, active_only=True)
        for mid in active_ids:
            ensure_model(mid)

    log.info("=== ensure_all_models completado ===")


# -- CLI ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model_sync",
        description="Sincroniza pesos de los modelos del release manifest "
                    "desde Google Drive a MODELS_DIR.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure_all", help="Sincroniza todos los modelos del manifest.")
    p_one = sub.add_parser("ensure", help="Sincroniza un solo modelo por id.")
    p_one.add_argument("model_id")
    sub.add_parser("verify",
                   help="Verifica integridad sin descargar (return code 0/1).")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        if args.cmd == "ensure_all":
            ensure_all_models()
        elif args.cmd == "ensure":
            ensure_model(args.model_id)
        elif args.cmd == "verify":
            manifest = get_manifest()
            bad = []
            for mid, mdata in manifest.get("models", {}).items():
                is_dep = mdata.get("deprecated", False)
                ok, issues = verify_extracted_files(mid)
                if is_dep and not ok:
                    log.info(f"  [{mid}] DEPRECATED, no descargado (esperado)")
                    continue
                status = "OK" if ok else f"FAIL ({len(issues)} issue/s)"
                log.info(f"  [{mid}] {status}")
                if not ok:
                    bad.append(mid)
            if bad:
                log.error(f"Modelos no verificados: {bad}")
                return 1
        return 0
    except ModelSyncError as e:
        log.error(f"model_sync abortado: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
