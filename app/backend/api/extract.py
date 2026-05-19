"""
Endpoint POST /api/extract-doc — Extracción de .doc legacy (Word 97-2003).

Convierte archivos en formato Composite Document File V2 (Word 97-2003) a
texto plano UTF-8 usando antiword como motor. Existe porque mammoth.js en
el frontend solo soporta .docx (Open XML) — los .doc binario requieren un
parser nativo.

DISEÑO DEFENSIVO:
  1. Pre-flight validation: tamaño <= 10 MB antes de tocar disco
  2. Streaming a NamedTemporaryFile (no acumula en RAM)
  3. Magic bytes check (D0 CF 11 E0 A1 B1 1A E1) antes de invocar antiword
  4. subprocess.run con timeout=30s y check=False
  5. Routing especifico de errores:
       - "is not a Word Document" en stderr  -> 400 Bad Request
       - TimeoutExpired                       -> 408 Request Timeout
       - FileNotFoundError (no antiword)      -> 503 Service Unavailable
       - exit != 0 sin match conocido         -> 500 Internal Server Error
       - exit == 0 + stdout vacio             -> 422 Unprocessable Entity
       - exit == 0 + stderr no vacio          -> 200 con partial=True
  6. Cleanup garantizado del tempfile en try/finally

Ver tests/test_extract_doc_endpoint.py para la validacion del comportamiento.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas import ExtractDocResponse

log = logging.getLogger(__name__)
router = APIRouter()


# Limite de tamano alineado con nginx client_max_body_size 10M
MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 MB

# Magic bytes del formato Composite Document File V2 (Microsoft OLE).
# Todo .doc legitimo de Word 97-2003 empieza con esta secuencia. Cualquier
# otra cosa (TXT, PDF, DOCX, binario aleatorio) NO matchea -> 400 inmediato
# sin llegar a invocar antiword.
CDF_V2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# Timeout maximo para la conversion. En benchmarks vimos < 10 ms para
# archivos legitimos hasta 10 MB; 30 s es margen 3000x para absorber
# load del sistema o algun edge case patologico no observado.
ANTIWORD_TIMEOUT_SEC = 30

# Stderr de antiword cuando el archivo NO es un .doc valido. Es 100%
# predictible y deterministico (lo verificamos con 6 tipos de archivos
# corruptos/falsificados). Si aparece -> 400 al cliente.
ANTIWORD_NOT_DOC_MARKER = "is not a Word Document"


@router.post("/api/extract-doc", response_model=ExtractDocResponse,
             tags=["Extraccion"])
async def extract_doc(file: UploadFile = File(...)) -> ExtractDocResponse:
    """
    Recibe un archivo .doc (Word 97-2003) y devuelve su texto plano en UTF-8.

    Errores posibles:
      - 400: archivo no es un .doc valido (magic bytes incorrectos o antiword
             rechaza el contenido)
      - 408: antiword excedio el timeout de 30 s
      - 413: archivo mayor a 10 MB
      - 422: antiword no extrajo ningun texto (documento vacio)
      - 500: error interno no clasificado de antiword
      - 503: antiword no esta instalado en el servidor (ver scripts/install_antiword.sh)
    """
    # 1) Leer todo el archivo a un tempfile, validando tamano en streaming
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            bytes_written = 0
            while True:
                chunk = await file.read(64 * 1024)  # 64 KB chunks
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_DOC_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Archivo demasiado grande "
                               f"(maximo {MAX_DOC_BYTES // (1024*1024)} MB).",
                    )
                tmp.write(chunk)

        if bytes_written == 0:
            raise HTTPException(
                status_code=400,
                detail="Archivo vacio. Subi un .doc valido de Word 97-2003.",
            )

        # 2) Validar magic bytes antes de invocar antiword (fast fail)
        with open(tmp_path, "rb") as f:
            header = f.read(8)
        if header[:8] != CDF_V2_MAGIC:
            raise HTTPException(
                status_code=400,
                detail="El archivo no es un .doc valido de Word 97-2003 "
                       "(magic bytes incorrectos). Si es un .docx convertilo "
                       "abriendolo en Word y guardando como .docx, o usa el "
                       "selector de archivos directamente con .docx.",
            )

        # 3) Invocar antiword con timeout y captura limpia
        try:
            result = subprocess.run(
                ["antiword", "-m", "UTF-8.txt", str(tmp_path)],
                capture_output=True, text=True,
                timeout=ANTIWORD_TIMEOUT_SEC,
                check=False,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            log.warning(f"antiword timeout en archivo de {bytes_written} bytes")
            raise HTTPException(
                status_code=408,
                detail=f"La conversion del .doc excedio el timeout de "
                       f"{ANTIWORD_TIMEOUT_SEC}s. El archivo puede estar danado.",
            )
        except FileNotFoundError:
            log.error("antiword NO esta instalado en el servidor. "
                      "Ver scripts/install_antiword.sh")
            raise HTTPException(
                status_code=503,
                detail="El servidor no tiene antiword instalado. "
                       "Contactar al administrador.",
            )

        # 4) Routing de errores segun exit code + stderr
        if result.returncode != 0:
            # Caso comun: archivo no es un .doc valido (TXT/PDF/DOCX/binario)
            if ANTIWORD_NOT_DOC_MARKER in (result.stderr or ""):
                raise HTTPException(
                    status_code=400,
                    detail="El archivo no es un .doc valido o esta demasiado "
                           "danado para extraer texto. Si es un .docx, usa el "
                           "selector con extension .docx directamente.",
                )
            # Otro error desconocido — log completo + 500 generico
            log.error(f"antiword fallo con exit={result.returncode}, "
                      f"stderr={result.stderr[:500]}")
            raise HTTPException(
                status_code=500,
                detail=f"Error interno al procesar el .doc "
                       f"(antiword exit={result.returncode}).",
            )

        # 5) exit == 0 — validar que haya texto
        text = (result.stdout or "").strip()
        if not text:
            raise HTTPException(
                status_code=422,
                detail="El .doc fue procesado pero no contiene texto extraible.",
            )

        # 6) Recuperacion parcial: exit 0 pero stderr no vacio = antiword
        # se salteo partes danadas. Avisamos al cliente para que sepa que
        # el texto puede estar incompleto.
        partial = bool((result.stderr or "").strip())
        warnings_msg = (result.stderr or "").strip()[:500] if partial else None

        return ExtractDocResponse(
            text=text,
            char_count=len(text),
            format="Carta",  # .doc no expone tamano de pagina facilmente
            partial=partial,
            warnings=warnings_msg,
        )

    finally:
        # Cleanup garantizado del tempfile en cualquier escenario
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
