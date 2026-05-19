"""
test_extract_doc_endpoint.py — Tests del endpoint POST /api/extract-doc.

Verifica el comportamiento defensivo del endpoint frente a:
  - Archivo .doc valido (fixture sample.doc)
  - Archivo vacio
  - Magic bytes invalidos (no es .doc real)
  - Archivo demasiado grande (> 10 MB)
  - Archivo .doc legitimamente corrupto (recuperacion parcial)
  - antiword no instalado en el sistema (mock con FileNotFoundError)
  - antiword timeout (mock con TimeoutExpired)

NOTA: solo `test_doc_valido_devuelve_texto` requiere antiword instalado.
El resto usa mocks o validacion de magic bytes (que se hace ANTES de
invocar antiword), asi que corren en cualquier entorno CI.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample.doc"


@pytest.fixture(scope="module")
def client():
    """Cliente FastAPI con solo el router de extract."""
    from backend.api import extract as extract_api
    app = FastAPI()
    app.include_router(extract_api.router)
    return TestClient(app)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _post_file(client, content: bytes, filename: str = "test.doc"):
    """Helper para POST multipart de un archivo con contenido custom."""
    return client.post(
        "/api/extract-doc",
        files={"file": (filename, io.BytesIO(content), "application/msword")},
    )


# ── Tests ────────────────────────────────────────────────────────────────────
def test_doc_valido_devuelve_texto(client):
    """Caso happy path: fixture sample.doc valido devuelve texto extraido."""
    if not FIXTURE_PATH.exists():
        pytest.skip("Fixture sample.doc no presente")
    # Si antiword no esta instalado en el CI, este test se salta limpio
    if not _antiword_available():
        pytest.skip("antiword no instalado (correr scripts/install_antiword.sh)")

    with open(FIXTURE_PATH, "rb") as f:
        resp = _post_file(client, f.read(), "sample.doc")

    assert resp.status_code == 200, f"esperado 200, recibido {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "text" in data
    assert "char_count" in data
    assert data["char_count"] > 0
    assert isinstance(data["text"], str)
    assert len(data["text"]) == data["char_count"]
    assert data["format"] == "Carta"
    # Validar UTF-8 valido (no bytes raw)
    data["text"].encode("utf-8")  # no lanza si es valido


def test_archivo_vacio_devuelve_400(client):
    """Archivo de 0 bytes -> 400 Bad Request."""
    resp = _post_file(client, b"", "empty.doc")
    assert resp.status_code == 400
    assert "vacio" in resp.json()["detail"].lower()


def test_magic_bytes_invalidos_devuelve_400(client):
    """Archivo con magic bytes incorrectos (no es CDF V2) -> 400."""
    # TXT renombrado como .doc
    resp = _post_file(client, b"hola mundo, soy un txt disfrazado", "fake.doc")
    assert resp.status_code == 400
    assert "magic bytes" in resp.json()["detail"].lower() or \
           "no es un .doc valido" in resp.json()["detail"].lower()


def test_docx_disfrazado_de_doc_devuelve_400(client):
    """DOCX (header PK = ZIP) renombrado a .doc -> 400 con sugerencia."""
    resp = _post_file(client, b"PK\x03\x04not a real docx content", "fake.doc")
    assert resp.status_code == 400
    # El mensaje debe sugerir usar .docx directamente
    assert ".docx" in resp.json()["detail"]


def test_archivo_demasiado_grande_devuelve_413(client):
    """Archivo > 10 MB -> 413 Payload Too Large."""
    # Construir un archivo de 11 MB con magic bytes validos al inicio
    # (el size check viene antes que el magic bytes check en el flujo)
    big = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * (11 * 1024 * 1024)
    resp = _post_file(client, big, "big.doc")
    assert resp.status_code == 413
    assert "10 mb" in resp.json()["detail"].lower() or \
           "demasiado grande" in resp.json()["detail"].lower()


def test_antiword_no_instalado_devuelve_503(client):
    """Si antiword no esta en el PATH del sistema -> 503 con mensaje claro."""
    if not FIXTURE_PATH.exists():
        pytest.skip("Fixture sample.doc no presente")

    with patch("backend.api.extract.subprocess.run",
               side_effect=FileNotFoundError("antiword")):
        with open(FIXTURE_PATH, "rb") as f:
            resp = _post_file(client, f.read(), "sample.doc")

    assert resp.status_code == 503
    assert "antiword" in resp.json()["detail"].lower()


def test_antiword_timeout_devuelve_408(client):
    """Si antiword tarda mas que ANTIWORD_TIMEOUT_SEC -> 408."""
    if not FIXTURE_PATH.exists():
        pytest.skip("Fixture sample.doc no presente")

    with patch("backend.api.extract.subprocess.run",
               side_effect=subprocess.TimeoutExpired(cmd="antiword", timeout=30)):
        with open(FIXTURE_PATH, "rb") as f:
            resp = _post_file(client, f.read(), "sample.doc")

    assert resp.status_code == 408
    assert "timeout" in resp.json()["detail"].lower()


def test_antiword_rechaza_archivo_con_magic_pasable(client):
    """Magic bytes validos PERO antiword rechaza el contenido -> 400.

    Caso raro pero posible: archivo con header CDF V2 correcto pero contenido
    interno corrupto que antiword identifica como 'is not a Word Document'.
    Lo simulamos con mock retornando exit=1 y stderr=ANTIWORD_NOT_DOC_MARKER.
    """
    fake_cdf = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1024

    fake_completed = subprocess.CompletedProcess(
        args=["antiword"], returncode=1,
        stdout="", stderr="fake.doc is not a Word Document.\n",
    )
    with patch("backend.api.extract.subprocess.run", return_value=fake_completed):
        resp = _post_file(client, fake_cdf, "fake.doc")

    assert resp.status_code == 400
    assert "no es un .doc valido" in resp.json()["detail"].lower() or \
           "danado" in resp.json()["detail"].lower()


def test_doc_corrupted_devuelve_200_con_partial_true(client):
    """exit 0 + stderr no vacio = antiword recupero contenido parcial.

    Comportamiento observado en el sandbox: si el .doc tiene bytes mutilados
    en el medio pero el header OLE es valido, antiword devuelve exit 0 con
    el texto que pudo extraer y un warning en stderr. El endpoint expone
    eso como `partial: true` para que el usuario sepa.
    """
    fake_cdf = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1024

    fake_completed = subprocess.CompletedProcess(
        args=["antiword"], returncode=0,
        stdout="texto extraido parcialmente del documento\n",
        stderr="Skipping an unmatched table row\n",
    )
    with patch("backend.api.extract.subprocess.run", return_value=fake_completed):
        resp = _post_file(client, fake_cdf, "corrupted.doc")

    assert resp.status_code == 200
    data = resp.json()
    assert data["partial"] is True
    assert data["warnings"] is not None
    assert "table row" in data["warnings"]
    assert data["text"]
    assert data["char_count"] > 0


def test_doc_sin_texto_devuelve_422(client):
    """exit 0 pero stdout vacio = documento sin texto extraible -> 422."""
    fake_cdf = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 1024

    fake_completed = subprocess.CompletedProcess(
        args=["antiword"], returncode=0, stdout="   \n  \n", stderr="",
    )
    with patch("backend.api.extract.subprocess.run", return_value=fake_completed):
        resp = _post_file(client, fake_cdf, "empty_doc.doc")

    assert resp.status_code == 422
    assert "texto" in resp.json()["detail"].lower()


# ── Helper de skip ───────────────────────────────────────────────────────────
def _antiword_available() -> bool:
    """Chequea si `antiword` esta en el PATH (para skippear tests que lo necesiten)."""
    import shutil
    return shutil.which("antiword") is not None
