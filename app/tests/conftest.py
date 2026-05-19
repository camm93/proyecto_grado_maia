"""
conftest.py — Fixtures compartidas para la suite de tests MAIA.

Diseño:
  - `synthetic_manifest`: dict en memoria con 2 modelos de juguete
    (mini files; hashes calculados al vuelo). Refleja la estructura
    real del release_manifest.json.
  - `models_dir`: tmp_path que actúa como MODELS_DIR; env var seteada.
  - `manifest_path`: tmp_path con el JSON serializado del manifest
    sintético; env var MANIFEST_PATH seteada.
  - `model_sync_reset`: limpia los caches del módulo entre tests.

Los tests NO deben requerir gdown ni torch instalados (todo está
mockeado), y NO deben tocar la red.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, content: bytes) -> str:
    """Escribe `content` a `path` y devuelve el SHA256 del archivo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return _sha256_bytes(content)


def _zip_dir(src_dir: Path, zip_path: Path, top_level: str | None = None) -> str:
    """Comprime `src_dir` a `zip_path`. Si `top_level` se pasa, los
    archivos quedan dentro de ese subdir dentro del ZIP. Devuelve SHA256
    del ZIP resultante.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    # Forzamos timestamp constante para que el SHA del ZIP sea estable
    # entre runs del mismo test (no entre tests distintos)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(src_dir.rglob("*")):
            if f.is_file():
                arc = f.relative_to(src_dir).as_posix()
                if top_level:
                    arc = f"{top_level}/{arc}"
                # ZipInfo con timestamp fijo para reproducibilidad
                info = zipfile.ZipInfo(arc, date_time=(2026, 5, 17, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                zf.writestr(info, f.read_bytes())
    return _sha256_bytes(zip_path.read_bytes())


# ── Fixtures principales ─────────────────────────────────────────────────────
@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Directorio temporal que actúa como MODELS_DIR."""
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(d))
    return d


@pytest.fixture
def synthetic_manifest(tmp_path):
    """Manifest dict + ZIPs reales en disco, con hashes consistentes.

    Construye dos modelos de juguete:
      - `mini-model-a`: 2 archivos required + 1 opcional
      - `mini-model-b`: 1 archivo required

    Devuelve dict con:
      - manifest: el dict JSON
      - zips: {model_id: Path al .zip}
      - extracted: {model_id: Path con árbol extraído de referencia}
    """
    workdir = tmp_path / "synthetic"
    zips_dir = workdir / "zips"
    extracted_root = workdir / "extracted"
    zips_dir.mkdir(parents=True)

    # ── Modelo A: 2 required + 1 opcional, ZIP wrapped (top_level) ───────────
    a_root = extracted_root / "mini-model-a"
    a_root.mkdir(parents=True)
    sha_config_a = _write(a_root / "config.json", b'{"model": "a"}\n')
    sha_weights_a = _write(a_root / "model.safetensors", b"FAKE-WEIGHTS-A" * 100)
    sha_optional_a = _write(a_root / "training_args.bin", b"trainer-metadata-a")
    zip_a = zips_dir / "mini-model-a.zip"
    sha_zip_a = _zip_dir(a_root, zip_a, top_level="mini-model-a")

    # ── Modelo B: 1 required, ZIP plano (sin top_level) ──────────────────────
    b_root = extracted_root / "mini-model-b"
    b_root.mkdir(parents=True)
    sha_config_b = _write(b_root / "config.json", b'{"model": "b"}\n')
    zip_b = zips_dir / "mini-model-b.zip"
    sha_zip_b = _zip_dir(b_root, zip_b, top_level=None)

    manifest = {
        "manifest_version": "test-1.0",
        "models": {
            "mini-model-a": {
                "task": "T2",
                "description": "Modelo de juguete A",
                "zip": {
                    "filename": "mini-model-a.zip",
                    "drive_url": "https://drive.google.com/file/d/FAKE_A/view",
                    "drive_file_id": "FAKE_A",
                    "size_mb": 1,
                    "sha256": sha_zip_a,
                },
                "extracted_dir": "mini-model-a",
                "files": [
                    {"path": "config.json", "sha256": sha_config_a,
                     "required": True},
                    {"path": "model.safetensors", "sha256": sha_weights_a,
                     "required": True},
                    {"path": "training_args.bin", "sha256": sha_optional_a,
                     "required": False},
                ],
            },
            "mini-model-b": {
                "task": "T1",
                "description": "Modelo de juguete B",
                "zip": {
                    "filename": "mini-model-b.zip",
                    "drive_url": "https://drive.google.com/file/d/FAKE_B/view",
                    "drive_file_id": "FAKE_B",
                    "size_mb": 1,
                    "sha256": sha_zip_b,
                },
                "extracted_dir": "mini-model-b",
                "files": [
                    {"path": "config.json", "sha256": sha_config_b,
                     "required": True},
                ],
            },
        },
        "llms": {
            "fake-llm": {"description": "No requiere descarga"},
        },
    }

    return {
        "manifest": manifest,
        "zips": {"mini-model-a": zip_a, "mini-model-b": zip_b},
        "extracted": {"mini-model-a": a_root, "mini-model-b": b_root},
    }


@pytest.fixture
def manifest_path(synthetic_manifest, tmp_path, monkeypatch):
    """Escribe el manifest sintético a disco y setea MANIFEST_PATH."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(synthetic_manifest["manifest"]), encoding="utf-8")
    monkeypatch.setenv("MANIFEST_PATH", str(p))
    return p


@pytest.fixture
def model_sync_reset(manifest_path, models_dir):
    """Importa model_sync y resetea sus caches para que vea las env
    vars frescas. Tests que tocan model_sync deben usar este fixture.
    """
    from backend.utils import model_sync
    model_sync.reset_caches()
    yield model_sync
    model_sync.reset_caches()
