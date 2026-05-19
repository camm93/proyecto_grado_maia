"""Esquemas Pydantic para los endpoints T1 y T2."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── T1: Segmentación retórica ────────────────────────────────────────────────
class SegmentRequest(BaseModel):
    text: str = Field(..., min_length=10,
                      description="Texto completo del artículo académico en español")
    model_id: str = Field(default="heuristic",
                          description="ID del modelo T1 a usar (ver GET /api/models?task=T1)")


class SegmentResult(BaseModel):
    index: int
    text: str
    char_start: int
    char_end: int
    word_count: int
    relative_pos: float
    rhetorical_zone: str
    t1_label: str
    t1_label_name: str
    t1_color: str
    t1_confidence: float
    low_confidence: bool


class SegmentResponse(BaseModel):
    model_id: str
    model_name: str
    mode: str
    elapsed_ms: int
    segments: list[SegmentResult]
    summary: dict   # total_segments, t1_distribution, avg_confidence


# ── T2: Detección de contribuciones ──────────────────────────────────────────
class FragmentInput(BaseModel):
    """Fragmento proveniente del output de /api/segment. Incluye contexto retórico."""
    index: int
    text: str
    t1_label: str = ""
    t1_label_name: str = ""
    t1_color: str = "#9CA3AF"
    t1_confidence: float = 0.0
    rhetorical_zone: str = ""
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    word_count: Optional[int] = None
    relative_pos: Optional[float] = None
    low_confidence: bool = False


class ContributionRequest(BaseModel):
    fragments: list[FragmentInput] = Field(
        ..., min_length=1,
        description="Segmentos clasificados por /api/segment (T1). "
                    "T2 usa el contexto retórico para enriquecer la detección.")
    model_id: str = Field(default="heuristic",
                          description="ID del modelo T2 a usar (ver GET /api/models?task=T2)")


class ContributionResult(BaseModel):
    index: int
    is_contribution: bool
    confidence: float
    label: str
    t1_label: str
    t1_label_name: str
    t1_color: str
    t1_confidence: float
    rhetorical_zone: str
    rhetorical_context: str
    char_start: Optional[int]
    char_end: Optional[int]
    word_count: Optional[int]
    relative_pos: Optional[float]


class ContributionResponse(BaseModel):
    model_id: str
    model_name: str
    mode: str
    elapsed_ms: int
    contributions: list[ContributionResult]
    summary: dict   # total, detected, rate, by_rhetorical_zone, by_t1_label


# ── Extracción de .doc legacy (Word 97-2003) ─────────────────────────────────
# Endpoint POST /api/extract-doc: convierte archivos .doc binario (Composite
# Document File V2) a texto plano usando antiword. Existe porque mammoth.js
# en el frontend solo soporta .docx (Open XML) — los .doc legacy requieren
# parser nativo. Ver v7.8 changelog y scripts/install_antiword.sh.
class ExtractDocResponse(BaseModel):
    text: str = Field(..., description="Texto extraído del documento en UTF-8")
    char_count: int = Field(..., description="Cantidad de caracteres extraídos")
    format: str = Field(default="Carta",
                        description="Formato de página detectado (Carta/A4/Oficio/Folio)")
    partial: bool = Field(default=False,
                          description="True si antiword recuperó contenido parcial "
                          "de un archivo con daños menores (stderr no vacío pero exit 0)")
    warnings: Optional[str] = Field(default=None,
                                    description="Mensaje de stderr de antiword si partial=True")
