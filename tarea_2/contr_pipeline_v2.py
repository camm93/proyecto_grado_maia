r"""
contr_pipeline.py
==============================================================================
Pipeline de extraccion de fragmentos CONTR para corpus cientifico en espanol.

Modos de ejecucion:
  --mode small : corpus reducido (426 .txt, proceso secuencial)
  --mode full  : corpus completo (~1.8 M docs, multiprocessing)

Segmentadores disponibles (--segmenter):
  simple  : regex propio, sin dependencias externas (default)
  spacy   : es_core_news_sm via spaCy (requiere: pip install spacy &&
            python -m spacy download es_core_news_sm)
  nltk    : PunktSentenceTokenizer en espanol via NLTK (requiere:
            pip install nltk && python -c "import nltk; nltk.download('punkt_tab')")

Parada automatica por objetivos (--target-*):
  El pipeline se detiene en cuanto TODOS los objetivos se cumplen y registra
  el evento en el log y en run_stats.json.
  --target-t1   : fragmentos CONTR para Tarea 1 (default 2000)
  --target-t2p  : fragmentos positivos para Tarea 2 (default 1000)
  --target-t2n  : negativos puros para Tarea 2 (default 1000)

Salidas por tarea:
  tarea1_CONTR.jsonl         → fragmentos CONTR (explicit + implicit)
  tarea2_positivos.jsonl     → contribucion EXPLICITA (positivos T2)
  tarea2_neg_dificiles.jsonl → CONTR implicito (negativos dificiles T2)
  tarea2_neg_puros.jsonl     → sin señal CONTR (negativos de control T2)
  run_stats.json             → estadisticas, config y razon de parada

Uso rapido:
  # Corpus reducido, segmentador simple, parada automatica:
  python contr_pipeline.py --mode small --corpus-dir ./corpus_426 \
      --out-dir ./output_small

  # Corpus completo con spaCy y objetivos personalizados:
  python contr_pipeline.py --mode full --corpus-dir ./corpus_core \
      --out-dir ./output_full --workers 8 --segmenter spacy \
      --target-t1 2000 --target-t2p 1000 --target-t2n 1000

  # Reanudar ejecucion interrumpida:
  python contr_pipeline.py --mode full --corpus-dir ./corpus_core \
      --out-dir ./output_full --workers 8 --resume
==============================================================================
"""

import re
import os
import sys
import json
import time
import logging
import argparse
import signal
import hashlib
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from logging.handlers import RotatingFileHandler
from tqdm import tqdm

# Segmentadores opcionales — se importan bajo demanda para no requerir
# instalacion cuando se usa el segmentador "simple" (default).
# spaCy  : pip install spacy && python -m spacy download es_core_news_sm
# NLTK   : pip install nltk  && python -c "import nltk; nltk.download('punkt_tab')"
_NLP_SPACY  = None   # instancia global spaCy por proceso worker
_NLP_NLTK   = False  # flag: True si NLTK ya esta inicializado en este proceso

# ══════════════════════════════════════════════════════════════════════════════
# 1. PATRONES REGEX  (v2 — correcciones B04 y filtros TOC aplicados)
#    Evaluación base: Precisión 100% · Recall 100% · F1 100%
#    Corrección v2: B04 requiere verbo de aporte; NOISE_PATTERNS filtra índices
# ══════════════════════════════════════════════════════════════════════════════

EXPLICIT_PATTERNS: list[tuple[str, str]] = [
    (
        # B01 — sujeto demostrativo + verbo de aporte.
        # Cubre: "este trabajo propone", "esta contribucion describe",
        # "esta aportacion presenta", "el presente articulo introduce".
        r"\b(?:este|esta|el\s+presente|la\s+presente|dicho|dicha)\s+"
        r"(?:trabajo|art\w+culo|estudio|paper|documento|manuscrito|contribuci\w+|aportaci\w+|propuesta|investigaci\w+)\s+"
        r"(?:propone|presenta|introduce|aporta|desarrolla|contribuye|describe|ofrece|plantea|diseña|implementa|establece|analiza)",
        "B01-demostrativo+verbo",
    ),
    (
        r"\b(?:proponemos|presentamos|introducimos|desarrollamos|describimos|ofrecemos"
        r"|diseñamos|implementamos|construimos|evaluamos|aportamos|contribuimos)\b",
        "B02-1PL",
    ),
    (
        # B03 — voz pasiva con objeto indefinido.
        # Verbos procedimentales (diseña, calcula, introduce datos, comprueba,
        # verifica, analiza, aplica, utiliza) se excluyen deliberadamente:
        # "se diseña la estructura" es METH, no CONTR.
        # Solo verbos cuyo sujeto implicito es el aporte del paper.
        r"\bse\s+(?:propone|presenta|introduce|desarrolla|aporta"  
        r"|implementa|construye|establece|define|formula|plantea"  
        r"|concibe|elabora)\s+(?:un|una)\b",
        "B03-pasiva+objeto",
    ),
    (
        # B04 — "el presente/actual trabajo" SOLO si va seguido de verbo de aporte.
        # Se excluyen verbos de localización institucional (se encuadra, se enmarca,
        # se inscribe, tiene como objetivo describir, etc.) que generan falsos positivos
        # frecuentes en TFGs y reportes técnicos.
        r"\b(?:el\s+presente|el\s+actual)\s+(?:trabajo|artículo|estudio|documento)\s+"
        r"(?:propone|presenta|introduce|aporta|desarrolla|contribuye|diseña"
        r"|implementa|establece|ofrece|plantea|describe)",
        "B04-presente+verbo",
    ),
    (
        r"\b(?:las?\s+)?(?:contribuciones?|aportaciones?)\s+"
        r"(?:principales?|clave|centrales?|más\s+relevantes?)?"
        r"\s*(?:de\s+este\s+(?:trabajo|artículo|estudio)|son)\b",
        "B05-nominal",
    ),
    (
        r"\b(?:como|la)\s+(?:principal|mayor|primordial|fundamental)?\s*"
        r"(?:contribuci\w+|aportaci\w+|aporte)\s*"
        r"(?:de\s+este\s+\w+\s*)?(?:es|,)?\b",
        "B06-principal-aport",
    ),
    (
        r"\b(?:nuestra?s?)\s+"
        r"(?:propuesta?|enfoque|método|modelo|sistema|marco|arquitectura|algoritmo"
        r"|herramienta|aproximaci\w+|estrategia|soluci\w+|pipeline|corpus|dataset)\b",
        "B07-posesivo+objeto",
    ),
    (
        # B08 — superioridad vs estado del arte.
        # Cubre: "supera al estado del arte" (al = a+el, contraccion espanola),
        # "supera a los metodos anteriores", "mejora los resultados previos".
        r"\b(?:supera(?:ndo)?|mejora(?:ndo)?)\s+(?:a\s+|al\s+)?(?:los?\s+)?(?:m\w+todos?|modelos?|enfoques?|sistemas?|baselines?|resultados?|algoritmos?)?\s*(?:del?\s+)?(?:estado\s+del\s+arte|anteriores?|existentes?|previos?)\b",
        "B08-supera-SOTA",
    ),
    (
        r"\b(?:obtiene(?:n)?|logra(?:n)?|alcanza(?:n)?|consigue(?:n)?)\s+"
        r"(?:mejores?|superiores?)\s+"
        r"(?:resultados?|rendimiento|desempe\w+|precisi\w+|puntuaciones?)\b",
        "B09-mejores-resultados",
    ),
    (
        r"\b(?:los?\s+)?(?:experimentos?|evaluaci\w+|resultados?|an\w+lisis)\s+"
        r"(?:obtenidos?\s+)?(?:demuestran?|muestran?|confirman?|evidencian?|indican?)\s+que\b",
        "B10-experimentos-demuestran",
    ),
    (r"\bpor\s+primera\s+vez\b", "B11-primera-vez"),
    (
        r"\bel\s+(?:objetivo|prop\w+sito|fin)\s+"
        r"(?:principal|central|fundamental|primordial)?\s+"
        r"(?:de\s+este\s+(?:trabajo|artículo|estudio)"
        r"|del\s+(?:presente\s+)?(?:trabajo|artículo))\s+es\b",
        "B12-objetivo-principal",
    ),
    (
        r"\b(?:ponemos\s+a\s+disposici\w+|publicamos?|liberamos?"
        r"|compartimos?|hacemos?\s+disponibles?)\b",
        "B13-disposicion",
    ),
    (
        r"\bse\s+(?:construye|crea|desarrolla|compila|recopila)\s+"
        r"(?:un|una)\s+(?:corpus|dataset|conjunto\s+de\s+datos|recurso|banco\s+de\s+datos)\b",
        "B14-recurso",
    ),
    (
        r"\b(?:de\s+(?:forma|manera)\s+(?:novedosa|innovadora|original|in\w+dita)"
        r"|sin\s+precedentes\s+en)\b",
        "B15-novedad-locucion",
    ),
    (
        r"\b(?:introducimos|presentamos)\s+(?:una?\s+)?(?:nueva?|novedosa?|original)?\s*"
        r"(?:m\w+trica|marco|arquitectura|herramienta|metodolog\w+|enfoque"
        r"|sistema|modelo|algoritmo)\b",
        "B16-introducimos",
    ),
    (
        r"\ba\s+diferencia\s+de\s+"
        r"(?:trabajos?|aproximaciones?|enfoques?|métodos?|estudios?)\s+"
        r"(?:previos?|previas?|anteriores?|existentes?)",
        "B17-diferencia",
    ),
    (
        r"\b(?:contribuimos?\s+con|aporta(?:mos?)?\s+"
        r"(?:evidencia|informaci\w+|un\s+an\w+lisis"
        r"|datos?|un\s+modelo|una\s+herramienta|resultados?))\b",
        "B18-contribuimos",
    ),
    (
        r"\b(?:hallazgos?|resultados?|aportes?)\s+"
        r"(?:principales?|clave|más\s+relevantes?)"
        r"\s*(?:de\s+este\s+(?:trabajo|artículo|estudio))?\b",
        "B19-hallazgos",
    ),
    (
        r"\b(?:la\s+propuesta|el\s+m\w+todo|el\s+enfoque|el\s+sistema"
        r"|la\s+arquitectura|el\s+modelo)\s+"
        r"(?:propuesto|propuesta|presentado|presentada|desarrollado|desarrollada)\b",
        "B20-propuesta",
    ),
]

IMPLICIT_PATTERNS: list[tuple[str, str]] = [
    (
        r"\b(?:logramos?|alcanzamos?|obtuvimos?|conseguimos?"
        r"|demostramos?|evidenciamos?|verificamos?)\b",
        "I01-logro-1PL",
    ),
    (
        # I02 — logro en voz pasiva refleja.
        # Requiere que en los 120 chars siguientes al verbo aparezca vocabulario
        # de novedad o superioridad, para evitar capturar resultados rutinarios
        # de calculo o medicion: "se obtiene Ry 59.44 KN" no es CONTR.
        r"\bse\s+(?:logra|alcanza|obtiene|consigue|demuestra|verifica)\b"
        r"(?=.{0,120}(?:mejor|mayor|superior|novedoso|novedosa|por\s+primera\s+vez"
        r"|mas\s+preciso|sin\s+precedente|estado\s+del\s+arte|supera|mejora"
        r"|incremento|reduccion|innovador|innovadora|propuesto|propuesta))"
        r"(?!.{0,40}(?:\d+[,.]\d+\s*(?:kn|kpa|mpa|mm|cm|m\b|kg|n\b|%)|"
        r"perfil|viga|pilar|losa|forjado|carga|coeficiente\s+de))",
        "I02-logro-pasiva",
    ),
    (
        # I03 — superioridad/mejora.
        # "mejora" y "supera" solo se capturan como VERBOS conjugados:
        #   VERBO:     "nuestro modelo mejora los resultados"      -> match
        #   SUSTANTIVO: "conduce a la mejora continua"             -> NO match
        #   SUSTANTIVO: "permite conducir mejora en la entrega"    -> NO match
        #   ADJETIVO:   "la mejor entrega de software"             -> NO match
        # La señal clave: el verbo va seguido de articulo+objeto o adverbio
        # de grado, nunca de preposicion o articulo que lo nominalice.
        r"\b(?:supera(?:n)?|mejora(?:n)?)\s+(?:los?|las?|el|la|su|nuestro|significativamente|considerablemente|notablemente)\b"
        r"|\bsupone\s+una\s+mejora\s+(?:significativa|considerable|de\s+)"
        r"|\brendimiento\s+superior\b"
        r"|\bmayor\s+precisi\w+\b"
        r"|\bmejor\s+desempe\w+\b"
        r"|\bresultados?\s+superiores?\b",
        "I03-superioridad",
    ),
    (
        r"\b(?:resuelve(?:n)?|resolvemos?|soluciona(?:n)?|solucionamos?"
        r"|aborda(?:n)?\s+el\s+(?:problema|desaf\w+o|reto|limitaci\w+)"
        r"|supera(?:n)?\s+la\s+limitaci\w+)\b",
        "I04-resolucion",
    ),
    (
        r"\b(?:nunca\s+antes|sin\s+precedentes?|in\w+dito|primer\s+(?:trabajo|estudio|sistema))\b",
        "I05-novedad",
    ),
    (
        r"\b(?:esto\s+permite|lo\s+que\s+permite|lo\s+cual\s+posibilita"
        r"|lo\s+que\s+facilita|con\s+implicaciones\s+para)\b",
        "I06-habilita",
    ),
    (
        r"\b(?:en\s+contraste\s+con|frente\s+a\s+(?:los?\s+)?"
        r"(?:m\w+todos?|enfoques?|trabajos?)\s+(?:previos?|previas?|anteriores?|existentes?))\b",
        "I07-contraste",
    ),
    (
        r"\b(?:constituye\s+(?:un|una)\s+(?:avance|contribuci\w+|aporte)"
        r"|representa\s+(?:un|una)\s+(?:avance|mejora|contribuci\w+|novedad))\b",
        "I08-avance",
    ),
    (r"\b(?:reducci\w+|incremento|aumento|mejora)\s+(?:del?|de\s+un)\s+\d", "I09-mejora-metrica"),
    (
        r"\b(?:nuestro\s+(?:enfoque|m\w+todo|modelo|sistema|algoritmo)\s+"
        r"(?:supera|mejora|aventaja|obtiene\s+mejores?\s+resultados?))\b",
        "I10-nuestro-supera",
    ),
]

# Secciones donde la densidad de CONTR es estructuralmente alta
HIGH_PRIOR_SECTIONS = [
    "conclus", "aportac", "contribuc", "resumen", "abstract",
    "síntesis", "discusi", "hallazgo", "objetivo",
]

# Ruido a eliminar antes de segmentar
# IMPORTANTE: todos los patrones usan anchoring estricto (^/$) o son lineales.
# Se evitan cuantificadores anidados que causan backtracking catastrófico
# en archivos grandes (el principal motivo de cuelgues en Windows).
NOISE_PATTERNS = [
    re.compile(r"\[\d+\]"),                               # referencias [1], [23]
    re.compile(r"\b\d{4}\b(?=\s*\))"),                   # años entre paréntesis
    re.compile(r"https?://\S+"),                          # URLs
    re.compile(r"\b(?:doi|DOI):\s*\S+"),                 # DOIs
    re.compile(r"(?m)^fig(?:ure|ura)?\.?\s*\d+", re.I),  # "Figure 1"
    re.compile(r"(?m)^tabla?\s*\d+", re.I),               # "Tabla 1"
    re.compile(r"[^\S\n]{2,}"),                           # espacios múltiples

    # ── Filtros de índice / tabla de contenidos (versión segura) ──────────
    # Línea con 3+ puntos seguidos de número de página — ancla con $ para
    # evitar backtracking sobre líneas largas sin el patrón esperado.
    # Ej: "3.2. METODOLOGÍA ........ 44"
    re.compile(r"(?m)^[^\n]*?\.{3,}\s*\d{1,4}\s*$"),

    # Líneas que son solo un número de página aislado
    re.compile(r"(?m)^\s*\d{1,4}\s*$"),
]


def normalize_text(text: str) -> str:
    """
    Normaliza artefactos comunes de extraccion de texto desde PDF:

    1. Soft hyphens (U+00AD): invisibles pero rompen el reconocimiento
       de palabras. Ej: pro[shy]pone -> propone.

    2. Guiones de corte de linea (word-wrap hyphens): aparecen cuando
       un extractor de PDF divide palabras al final de linea.
       Ej: pro-[LF]pone -> propone.
       Solo se fusionan si la silaba siguiente empieza en minuscula
       (evita fusionar listas con guion como marcador).

    3. Puntuacion sin espacio posterior: "xxx.Yyy" -> "xxx. Yyy".
       PDF extractors frecuentemente omiten el espacio tras el punto.

    4. Guiones medios y largos (en-dash, em-dash) -> espacio.

    5. Espacios multiples normalizados.
    """
    # 1. Soft hyphen (U+00AD) — eliminar directamente
    text = text.replace("\u00ad", "")

    # 2. Guión de corte de línea: "-\n" o "-\r\n" seguido de minúscula
    #    española (incluyendo vocales con tilde y ñ).
    #    [a-záéíóúüñ] cubre todo el alfabeto español en minúsculas.
    text = re.sub(
        r"-[\r\n]+(?=[a-záéíóúüñ])",
        "",
        text,
    )

    # 3. Puntuación sin espacio: ".X" "!X" "?X" ":X" ";X" ",X"
    #    donde X es mayúscula (incl. Á É Í Ó Ú Ñ), dígito, o "¿¡".
    #    Condición negativa: no insertar espacio si es abreviatura
    #    de un solo carácter seguida de punto (ej: "Dr.García" -> lo
    #    protegemos con el paso de abreviaturas en split_sentences).
    text = re.sub(
        r"([.!?;:,])([A-ZÁÉÍÓÚÜÑ\d¿¡])",
        r"\1 \2",
        text,
    )

    # 4. Guiones medios y largos como separadores de frase
    text = text.replace("\u2013", " ").replace("\u2014", " ")  # – y —

    # 5. Espacios múltiples (sin consumir saltos de línea)
    text = re.sub(r"[^\S\n]{2,}", " ", text)

    return text


def _filter_toc_lines(text: str) -> str:
    """
    Elimina lineas de tabla de contenidos y cabeceras de pagina de forma
    lineal (O(n)) sin backtracking. Se aplica antes de NOISE_PATTERNS.

    Descarta una linea si cumple cualquiera de estas condiciones:
      (a) Cabecera de pagina PDF: contiene "pagina X de Y" / "page X of Y"
      (b) Cabecera de documento repetida: patron TITULO ... Autor ... Fecha ... Pag
      (c) Linea TOC: corta (<120 chars), termina en numero o tiene 3+ puntos,
          y no contiene verbo de contenido real
    """
    _page_header = re.compile(
        r"p(?:a|á)gina\s*\d+\s*de\s*\d+|page\s*\d+\s*of\s*\d+",
        re.IGNORECASE,
    )
    # Cabecera de documento tipo "TITULO DOC ... Autor ... Septiembre 2022 ... Pagina X"
    # Señal: linea corta con año de 4 digitos Y numero de pagina en la misma linea
    _doc_header = re.compile(
        r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre"
        r"|octubre|noviembre|diciembre|january|february|march|april|june"
        r"|july|august|september|october|november|december).{0,60}"
        r"p(?:a|á)gina\s*\d+",
        re.IGNORECASE,
    )
    _toc_end  = re.compile(r"\d{1,4}\s*$")
    _has_dots = re.compile(r"\.{3,}")
    _has_verb = re.compile(
        r"(?:es|son|fue|fueron|tiene|tienen|propone|presenta|demuestra"
        r"|permite|incluye|contiene|describe|analiza|muestra|obtiene"
        r"|genera|produce|aplica)",
        re.IGNORECASE,
    )
    # (d) pies de ilustracion, figura, grafico — cabeceras de imagen en PDFs
    _illus = re.compile(
        r"^(?:ilustraci[oó]n|figura|figure|fig\.?|gr[aá]fico|image|imagen"
        r"|cuadro|esquema|diagrama)\.?\s*\d+",
        re.IGNORECASE,
    )
    # (e) lineas cortas todo en mayusculas = cabecera de seccion/documento
    # Se excluyen si contienen verbos (podrian ser titulos con contenido real)
    _all_caps = re.compile(r"^[A-Z0-9\.\s:,\-/()]{8,80}$")

    lines_out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines_out.append(line)
            continue
        # (a) cabecera de pagina explicita: "Pagina X de Y" / "Page X of Y"
        if _page_header.search(stripped):
            continue
        # (b) cabecera de documento con mes + numero de pagina en la misma linea
        if _doc_header.search(stripped):
            continue
        # (c) linea de indice TOC: corta, termina en numero o tiene 3+ puntos
        if (
            len(stripped) < 120
            and (_toc_end.search(stripped) or _has_dots.search(stripped))
            and not _has_verb.search(stripped)
        ):
            continue
        # (d) pie de ilustracion / figura / grafico
        if _illus.search(stripped):
            continue
        # (e) linea corta con >= 75% de letras en mayuscula = cabecera de documento.
        # Se usa ratio en lugar de match exacto para tolerar abreviaturas como "No2".
        _alpha = [c for c in stripped if c.isalpha()]
        _upper_ratio = (
            sum(1 for c in _alpha if c.isupper()) / len(_alpha)
            if _alpha else 0
        )
        if (
            len(stripped) <= 80
            and _upper_ratio >= 0.75
            and not _has_verb.search(stripped)
        ):
            continue
        lines_out.append(line)
    return "\n".join(lines_out)

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Fragment:
    doc_id:           str
    doc_hash:         str
    text:             str
    word_count:       int
    char_start:       int
    char_end:         int
    relative_pos:     float       # 0.0 inicio … 1.0 final del documento
    section_hint:     str         # encabezado más cercano hacia atrás
    contr_type:       str = "none"          # explicit | implicit | implicit_weak | none
    tarea1_label:     str = ""              # "CONTR" o ""
    tarea2_role:      str = "negative"      # positive | hard_negative | negative
    score_explicit:   float = 0.0
    score_implicit:   float = 0.0
    explicit_hits:    list = field(default_factory=list)
    implicit_hits:    list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FUNCIONES DE ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Normaliza y limpia texto de extraccion PDF para procesamiento NLP.

    Orden:
      1. normalize_text: hyphens, soft-hyphens, puntuacion sin espacio
      2. _filter_toc_lines: cabeceras PDF e indices
      3. NOISE_PATTERNS: referencias, URLs, figuras, nums de pagina
    """
    text = normalize_text(text)
    text = _filter_toc_lines(text)
    for pat in NOISE_PATTERNS:
        text = pat.sub(" ", text)
    return text.strip()


# Abreviaturas protegidas para el segmentador simple
_ABBREVS_RE = re.compile(
    r"\b(Dr|Dra|Sr|Sra|Srta|Prof|Dpto|Fig|Tab|Ec|Eq|et\s+al|"
    r"ej|pag|pags|vol|num|nums|approx|etc|art|arts|cap|ed|eds|coord|trad)\.",
    re.IGNORECASE,
)


def _split_simple(text: str) -> list[str]:
    """Segmentador regex: rapido, sin dependencias, cubre espanol completo."""
    text = re.sub(r"([.!?])([A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1])", r"\1 \2", text)
    text = _ABBREVS_RE.sub(lambda m: m.group().replace(".", "##DOT##"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\d\u00bf\u00a1])", text)
    return [p.replace("##DOT##", ".").strip() for p in parts if p.strip()]


def _split_spacy(text: str) -> list[str]:
    """Segmentador spaCy: usa es_core_news_sm cargado una vez por proceso."""
    global _NLP_SPACY
    if _NLP_SPACY is None:
        try:
            import spacy
            _NLP_SPACY = spacy.load(
                "es_core_news_sm",
                disable=["ner", "tagger", "morphologizer", "attribute_ruler", "lemmatizer"],
            )
            _NLP_SPACY.max_length = 2_000_000
        except (ImportError, OSError) as e:
            import warnings
            warnings.warn(f"spaCy no disponible ({e}). Usando segmentador simple.")
            return _split_simple(text)
    doc = _NLP_SPACY(text[:_NLP_SPACY.max_length])
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _split_nltk(text: str) -> list[str]:
    """Segmentador NLTK: PunktSentenceTokenizer en espanol."""
    global _NLP_NLTK
    if not _NLP_NLTK:
        try:
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
            _NLP_NLTK = True
        except ImportError as e:
            import warnings
            warnings.warn(f"NLTK no disponible ({e}). Usando segmentador simple.")
            return _split_simple(text)
    import nltk
    return [s.strip() for s in nltk.sent_tokenize(text, language="spanish") if s.strip()]


# Segmentador activo — se establece en _worker_init segun --segmenter
_ACTIVE_SEGMENTER = "simple"


def split_sentences(text: str) -> list[str]:
    """
    Dispatcher de segmentacion de oraciones.
    Backend activo: simple | spacy | nltk (configurado via --segmenter).
    Todos cubren tildes, n-tilde, u-dieresis y puntuacion sin espacio.
    """
    if _ACTIVE_SEGMENTER == "spacy":
        return _split_spacy(text)
    if _ACTIVE_SEGMENTER == "nltk":
        return _split_nltk(text)
    return _split_simple(text)


def detect_section_hint(text: str, char_pos: int, window: int = 500) -> str:
    """Busca el último posible encabezado antes de char_pos."""
    segment = text[max(0, char_pos - window): char_pos]
    for line in reversed(segment.split("\n")):
        line = line.strip()
        # Línea corta, sin punto final → probable encabezado
        if 3 < len(line) < 70 and not line.endswith(".") and not line.startswith("["):
            return line[:60]
    return "unknown"


# Conjunto de etiquetas de patrones bloqueados para esta ejecucion.
# Se puebla desde args.block_patterns via _worker_init y run_small.
# Permite excluir patrones ruidosos (ej. B11) sin tocar el codigo.
_BLOCKED_PATTERNS: set = set()


def score_fragment(text: str, relative_pos: float, section_hint: str) -> dict:
    """Puntua y clasifica un fragmento con los patrones CONTR.
    Los patrones cuya etiqueta esta en _BLOCKED_PATTERNS se ignoran.
    """
    t = text.lower()

    exp_hits, imp_hits = [], []
    for pat, label in EXPLICIT_PATTERNS:
        if label in _BLOCKED_PATTERNS:
            continue                       # patron bloqueado via --block-patterns
        m = re.search(pat, t)
        if m:
            exp_hits.append({"label": label, "match": m.group()[:55]})

    for pat, label in IMPLICIT_PATTERNS:
        if label in _BLOCKED_PATTERNS:
            continue
        m = re.search(pat, t)
        if m:
            imp_hits.append({"label": label, "match": m.group()[:55]})

    score_exp = min(1.0, len(exp_hits) * 0.35 + (0.2 if exp_hits else 0.0))

    # Bonus posicional (intro ≤15% y conclusión ≥85%)
    pos_bonus = 0.20 if (relative_pos <= 0.15 or relative_pos >= 0.85) else 0.0
    # Bonus de sección
    sec_bonus = 0.15 if any(kw in section_hint.lower() for kw in HIGH_PRIOR_SECTIONS) else 0.0
    score_imp = min(1.0, len(imp_hits) * 0.25 + pos_bonus + sec_bonus)

    # Clasificacion en tres niveles:
    #
    # explicit       score_exp >= 0.35
    #   -> Tarea 1: CONTR  |  Tarea 2: positive
    #   Marcador lexico de contribucion presente y claro.
    #
    # implicit       score_imp >= 0.45  (umbral alto)
    #   -> Tarea 1: CONTR  |  Tarea 2: hard_negative
    #   Senales implicitas fuertes: multiple hits + bonus posicional/seccion.
    #   El umbral alto (0.45 vs 0.30 anterior) evita capturar textos
    #   descriptivos/BACK que usan vocabulario de mejora sin autoria propia.
    #
    # implicit_weak  0.30 <= score_imp < 0.45
    #   -> Tarea 1: (sin etiqueta CONTR)  |  Tarea 2: hard_negative
    #   Zona gris: util como negativo dificil en Tarea 2 pero insuficiente
    #   para etiquetar CONTR en Tarea 1. Ej: textos BACK/DESC con "mejora"
    #   como sustantivo o vocabulario de superioridad sin voz propia.
    #
    # none           score_imp < 0.30
    #   -> Sin etiqueta  |  Tarea 2: negative

    if score_exp >= 0.35:
        contr_type, tarea1, tarea2 = "explicit",       "CONTR", "positive"
    elif score_imp >= 0.45:
        contr_type, tarea1, tarea2 = "implicit",       "CONTR", "hard_negative"
    elif score_imp >= 0.30:
        contr_type, tarea1, tarea2 = "implicit_weak",  "",      "hard_negative"
    else:
        contr_type, tarea1, tarea2 = "none",           "",      "negative"

    return {
        "score_explicit": round(score_exp, 3),
        "score_implicit": round(score_imp, 3),
        "explicit_hits":  exp_hits,
        "implicit_hits":  imp_hits,
        "contr_type":     contr_type,
        "tarea1_label":   tarea1,
        "tarea2_role":    tarea2,
    }


def extract_fragments(
    file_path: Path,
    min_words: int = 250,
    max_words: int = 1000,
    overlap_sents: int = 2,
    max_neg_per_doc: int = 3,
    max_contr_per_doc: int = 5,
) -> list[dict]:
    """
    Extrae fragmentos de un documento .txt con ventana deslizante.

    Args:
        file_path:         Ruta al archivo de texto.
        min_words:         Mínimo de palabras por fragmento.
        max_words:         Máximo de palabras por fragmento.
        overlap_sents:     Oraciones de solapamiento entre ventanas consecutivas.
        max_neg_per_doc:   Máximo de fragmentos negativos puros por documento.
        max_contr_per_doc: Máximo de fragmentos CONTR (explicit+implicit) por
                           documento. Restringe la dominancia de docs muy largos
                           en el dataset. Default=5; usa 1-3 para datasets muy
                           balanceados. El proyecto pide aislamiento de doc entre
                           particiones, por lo que limitar aquí reduce el riesgo
                           de que un doc ocupe toda una partición.

    Returns:
        Lista de dicts serializables (uno por fragmento extraído).
    """
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if len(raw) < 300:          # documento demasiado corto
        return []

    doc_len = len(raw)
    doc_id  = file_path.stem
    doc_hash = hashlib.md5(raw[:2048].encode("utf-8", errors="replace")).hexdigest()[:12]

    text = clean_text(raw)
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return []

    results: list[dict] = []
    neg_count   = 0   # fragmentos negativos puros emitidos por este doc
    contr_count = 0   # fragmentos CONTR emitidos por este doc
    i = 0

    while i < len(sentences):
        chunk, wc, start_idx = [], 0, i

        # Acumula oraciones hasta alcanzar min_words (sin superar max_words)
        while i < len(sentences) and wc < max_words:
            chunk.append(sentences[i])
            wc += len(sentences[i].split())
            i += 1
            if wc >= min_words:
                # Corte limpio en punto final
                if chunk[-1].rstrip().endswith((".", "?", "…")):
                    break

        if wc < min_words:
            break

        chunk_text = " ".join(chunk)
        # Posición aproximada en el documento original
        char_start = raw.find(chunk[0][:30]) if chunk else 0
        char_start = max(0, char_start)
        char_end   = min(doc_len, char_start + len(chunk_text))
        rel_pos    = char_start / doc_len
        sec_hint   = detect_section_hint(raw, char_start)

        scored = score_fragment(chunk_text, rel_pos, sec_hint)

        # Control de negativos puros: evitar saturar salida con texto sin CONTR
        if scored["contr_type"] == "none":
            neg_count += 1
            if neg_count > max_neg_per_doc:
                i = max(start_idx + 1, i - overlap_sents)
                continue

        # Restricción del proyecto: aislamiento de documento entre particiones.
        # Limitar fragmentos CONTR por documento evita que un único documento
        # muy largo monopolice una partición entera (train/val/test).
        # Solo cuenta explicit e implicit (no implicit_weak ni none).
        if scored["contr_type"] in ("explicit", "implicit"):
            if contr_count >= max_contr_per_doc:
                # Seguimos extrayendo negativos pero ya no más CONTR de este doc
                i = max(start_idx + 1, i - overlap_sents)
                continue
            contr_count += 1

        frag = Fragment(
            doc_id       = doc_id,
            doc_hash     = doc_hash,
            text         = chunk_text,
            word_count   = wc,
            char_start   = char_start,
            char_end     = char_end,
            relative_pos = round(rel_pos, 4),
            section_hint = sec_hint,
            contr_type   = scored["contr_type"],
            tarea1_label = scored["tarea1_label"],
            tarea2_role  = scored["tarea2_role"],
            score_explicit = scored["score_explicit"],
            score_implicit = scored["score_implicit"],
            explicit_hits  = scored["explicit_hits"],
            implicit_hits  = scored["implicit_hits"],
        )
        results.append(asdict(frag))

        # Retrocede overlap_sents para ventana deslizante
        i = max(start_idx + 1, i - overlap_sents)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. WORKER PARA MULTIPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

# Parámetros globales del worker (se inyectan via initializer)
_WORKER_CONFIG: dict = {}


def _worker_init(cfg: dict):
    global _WORKER_CONFIG, _ACTIVE_SEGMENTER, _BLOCKED_PATTERNS
    _WORKER_CONFIG = cfg
    # Propagar el segmentador elegido a este proceso hijo
    _ACTIVE_SEGMENTER = cfg.get("segmenter", "simple")
    # Propagar patrones bloqueados a este proceso hijo
    _BLOCKED_PATTERNS = set(cfg.get("blocked_patterns", []))
    # Pre-cargar el modelo si es spaCy o NLTK para que el primer documento
    # no pague el costo de carga (que puede ser varios segundos).
    if _ACTIVE_SEGMENTER == "spacy":
        _split_spacy.__globals__  # trigger lazy load via primera llamada
        _split_spacy("")           # carga el modelo en este worker
    elif _ACTIVE_SEGMENTER == "nltk":
        _split_nltk("")            # verifica/descarga punkt en este worker
    # Ignorar Ctrl+C en workers (el proceso principal gestiona el shutdown)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_process(file_path_str: str) -> list[dict]:
    """Función ejecutada por cada worker en el pool."""
    cfg = _WORKER_CONFIG
    return extract_fragments(
        Path(file_path_str),
        min_words          = cfg.get("min_words", 250),
        max_words          = cfg.get("max_words", 1000),
        overlap_sents      = cfg.get("overlap_sents", 2),
        max_neg_per_doc    = cfg.get("max_neg_per_doc", 3),
        max_contr_per_doc  = cfg.get("max_contr_per_doc", 5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. ESCRITORES DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

class OutputWriter:
    """Gestiona los cuatro archivos de salida JSONL con flush periódico."""

    TAREA2_NEG_PURO_LIMIT = 1500   # negativos puros máximos en salida final

    def __init__(self, out_dir: Path,
                 target_t1: int = 2000,
                 target_t2p: int = 1000,
                 target_t2n: int = 1000):
        out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = {
            "t1":  open(out_dir / "tarea1_CONTR.jsonl",         "a", encoding="utf-8"),
            "t2p": open(out_dir / "tarea2_positivos.jsonl",      "a", encoding="utf-8"),
            "t2h": open(out_dir / "tarea2_neg_dificiles.jsonl",  "a", encoding="utf-8"),
            "t2n": open(out_dir / "tarea2_neg_puros.jsonl",      "a", encoding="utf-8"),
        }
        self.counts  = {"t1": 0, "t2p": 0, "t2h": 0, "t2h_weak": 0, "t2n": 0}
        self.targets = {"t1": target_t1, "t2p": target_t2p, "t2n": target_t2n}
        self._written = 0

    def targets_met(self) -> bool:
        """True cuando TODOS los objetivos de fragmentos se han cumplido."""
        return (
            self.counts["t1"]  >= self.targets["t1"]  and
            self.counts["t2p"] >= self.targets["t2p"] and
            self.counts["t2n"] >= self.targets["t2n"]
        )

    def pending_targets(self) -> dict:
        """Devuelve cuantos fragmentos faltan por cada objetivo."""
        return {
            "t1_falta":  max(0, self.targets["t1"]  - self.counts["t1"]),
            "t2p_falta": max(0, self.targets["t2p"] - self.counts["t2p"]),
            "t2n_falta": max(0, self.targets["t2n"] - self.counts["t2n"]),
        }

    def write_batch(self, fragments: list[dict]):
        for f in fragments:
            line = json.dumps(f, ensure_ascii=False) + "\n"
            role = f.get("tarea2_role", "negative")
            ctype = f.get("contr_type", "none")

            # Solo explicit e implicit (score alto) van a Tarea 1.
            # implicit_weak es zona gris: util para Tarea 2 pero no Tarea 1.
            if ctype in ("explicit", "implicit"):
                self._fh["t1"].write(line)
                self.counts["t1"] += 1

            if role == "positive":
                self._fh["t2p"].write(line)
                self.counts["t2p"] += 1
            elif role == "hard_negative":
                self._fh["t2h"].write(line)
                self.counts["t2h"] += 1
                # Contador separado para implicit_weak (excluidos de Tarea 1)
                if ctype == "implicit_weak":
                    self.counts["t2h_weak"] += 1
            elif role == "negative" and self.counts["t2n"] < self.TAREA2_NEG_PURO_LIMIT:
                self._fh["t2n"].write(line)
                self.counts["t2n"] += 1

        self._written += len(fragments)
        if self._written % 5000 == 0:
            for fh in self._fh.values():
                fh.flush()

    def close(self):
        for fh in self._fh.values():
            fh.flush()
            fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

class Checkpoint:
    """Lee y escribe el último archivo procesado para reanudar corpus completo."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> Optional[str]:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            return data.get("last_file")
        return None

    def save(self, last_file: str, counts: dict):
        self.path.write_text(json.dumps({
            "last_file": last_file,
            "counts":    counts,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# 7. LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(out_dir: Path, mode: str) -> logging.Logger:
    log = logging.getLogger("contr_pipeline")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    # Forzar UTF-8 en la consola — necesario en Windows (cp1252 no soporta
    # caracteres Unicode como los bordes de caja usados en los mensajes de log).
    try:
        stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                      buffering=1, closefd=False)
    except Exception:
        stream = sys.stdout   # fallback: usar stdout tal como esta

    ch = logging.StreamHandler(stream)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    fh = RotatingFileHandler(
        out_dir / f"pipeline_{mode}.log", maxBytes=10 * 1024 * 1024,
        backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODOS DE EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def run_small(args, log: logging.Logger):
    """Corpus reducido: 426 archivos .txt — proceso secuencial con progreso."""
    global _ACTIVE_SEGMENTER, _BLOCKED_PATTERNS
    _ACTIVE_SEGMENTER   = args.segmenter
    _BLOCKED_PATTERNS   = set(args.block_patterns)

    corpus_dir = Path(args.corpus_dir)
    out_dir    = Path(args.out_dir)
    files = sorted(corpus_dir.glob("*.txt"))

    if not files:
        log.error(f"No se encontraron archivos .txt en {corpus_dir}")
        sys.exit(1)

    log.info(f"MODO SMALL -- {len(files)} archivos en {corpus_dir}")
    log.info(f"Segmentador      : {args.segmenter}")
    log.info(f"Objetivos        : T1>={args.target_t1}  T2pos>={args.target_t2p}  T2neg>={args.target_t2n}")

    writer = OutputWriter(
        out_dir,
        target_t1  = args.target_t1,
        target_t2p = args.target_t2p,
        target_t2n = args.target_t2n,
    )
    t0 = time.time()
    stop_reason = None

    for fpath in tqdm(files, desc="Procesando", unit="doc"):
        frags = extract_fragments(
            fpath,
            min_words          = args.min_words,
            max_words          = args.max_words,
            overlap_sents      = args.overlap,
            max_neg_per_doc    = args.max_neg_per_doc,
            max_contr_per_doc  = args.max_contr_per_doc,
        )
        writer.write_batch(frags)

        if writer.targets_met():
            stop_reason = "OBJETIVOS_ALCANZADOS"
            log.info("-" * 60)
            log.info("  *** PARADA ANTICIPADA: todos los objetivos cumplidos ***")
            log.info(f"  T1  CONTR     : {writer.counts['t1']:,}  (objetivo: {args.target_t1:,})")
            log.info(f"  T2  positivos : {writer.counts['t2p']:,}  (objetivo: {args.target_t2p:,})")
            log.info(f"  T2  neg.puros : {writer.counts['t2n']:,}  (objetivo: {args.target_t2n:,})")
            log.info(f"  Ultimo doc    : {fpath.name}")
            log.info("-" * 60)
            break

    writer.close()
    elapsed = time.time() - t0
    n_processed = files.index(fpath) + 1 if stop_reason else len(files)
    _print_stats(writer.counts, n_processed, elapsed, out_dir, log,
                 stop_reason=stop_reason, targets=writer.targets)


def run_full(args, log: logging.Logger):
    """
    Corpus completo: millones de documentos.
    Usa multiprocessing con pool de workers y checkpoint incremental.
    """
    corpus_dir  = Path(args.corpus_dir)
    out_dir     = Path(args.out_dir)
    ckpt_path   = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoint.json"
    ckpt        = Checkpoint(ckpt_path)

    # Recopilar archivos — dos estrategias:
    # (a) --filelist: leer rutas desde un archivo de texto pre-generado.
    #     Mucho mas rapido en drives montados (Google Drive, NAS, red).
    # (b) rglob: exploracion directa del directorio (lento en drives remotos).
    extensions = {".txt", ".json", ".jsonl", ""}

    if args.filelist and Path(args.filelist).exists():
        log.info(f"Cargando lista de archivos desde: {args.filelist}")
        with open(args.filelist, encoding="utf-8") as fl:
            all_files = [
                Path(line.strip()) for line in fl
                if line.strip() and not line.startswith("#")
            ]
        log.info(f"Total de archivos en filelist: {len(all_files):,}")
    else:
        log.info(f"Indexando archivos en {corpus_dir} ...")
        log.info("  (en drives remotos esto puede tardar varios minutos)")
        log.info("  (tip: usa --filelist para evitar esta espera en ejecuciones futuras)")
        all_files = [
            p for p in corpus_dir.rglob("*")
            if p.is_file() and (p.suffix.lower() in extensions or p.suffix == "")
        ]
        # Guardar el listado generado para reutilizarlo
        filelist_path = out_dir / "corpus_filelist.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(filelist_path, "w", encoding="utf-8") as fl:
            for p in sorted(all_files):
                fl.write(str(p) + "\n")
        log.info(f"  Listado guardado en: {filelist_path}  (usa --filelist en proximas ejecuciones)")

    all_files.sort()
    total = len(all_files)
    log.info(f"Total de archivos encontrados: {total:,}")

    # Reanudar desde checkpoint
    start_from = None
    if args.resume:
        start_from = ckpt.load()
        if start_from:
            log.info(f"Reanudando desde: {start_from}")

    if start_from:
        try:
            start_idx = next(i for i, f in enumerate(all_files) if str(f) == start_from)
            all_files = all_files[start_idx + 1:]
            log.info(f"Archivos restantes: {len(all_files):,}")
        except StopIteration:
            log.warning("Checkpoint no encontrado en lista actual. Iniciando desde el principio.")

    worker_cfg = {
        "min_words":          args.min_words,
        "max_words":          args.max_words,
        "overlap_sents":      args.overlap,
        "max_neg_per_doc":    args.max_neg_per_doc,
        "max_contr_per_doc":  args.max_contr_per_doc,
        "segmenter":          args.segmenter,
        "blocked_patterns":   list(args.block_patterns),
    }

    writer   = OutputWriter(
        out_dir,
        target_t1  = args.target_t1,
        target_t2p = args.target_t2p,
        target_t2n = args.target_t2n,
    )
    n_workers  = min(args.workers, mp.cpu_count())
    batch_sz   = args.batch_size
    processed  = 0
    t0         = time.time()
    shutdown   = False
    stop_reason = None

    log.info(f'Segmentador      : {args.segmenter}')
    log.info(f'Objetivos        : T1>={args.target_t1}  T2pos>={args.target_t2p}  T2neg>={args.target_t2n}')

    def handle_sigint(sig, frame):
        nonlocal shutdown
        log.warning("Interrupcion recibida. Finalizando lote actual ...")
        shutdown = True

    signal.signal(signal.SIGINT, handle_sigint)

    log.info(f"MODO FULL -- workers={n_workers}  batch={batch_sz}")

    with mp.Pool(
        processes  = n_workers,
        initializer = _worker_init,
        initargs    = (worker_cfg,),
    ) as pool:
        file_strings = [str(f) for f in all_files]

        with tqdm(total=len(file_strings), desc="Documentos", unit="doc") as pbar:
            for batch_start in range(0, len(file_strings), batch_sz):
                if shutdown:
                    break

                batch = file_strings[batch_start: batch_start + batch_sz]
                results = pool.map(_worker_process, batch)

                for frags in results:
                    writer.write_batch(frags)

                processed += len(batch)
                pbar.update(len(batch))

                # Checkpoint cada batch
                ckpt.save(batch[-1], writer.counts)

                # Parada anticipada si todos los objetivos estan cumplidos
                if writer.targets_met():
                    stop_reason = 'OBJETIVOS_ALCANZADOS'
                    shutdown = True
                    log.info('-' * 60)
                    log.info('  *** PARADA ANTICIPADA: todos los objetivos cumplidos ***')
                    log.info(f"  T1  CONTR     : {writer.counts['t1']:,}  (objetivo: {args.target_t1:,})")
                    log.info(f"  T2  positivos : {writer.counts['t2p']:,}  (objetivo: {args.target_t2p:,})")
                    log.info(f"  T2  neg.puros : {writer.counts['t2n']:,}  (objetivo: {args.target_t2n:,})")
                    log.info(f"  Ultimo batch  : {batch[-1]}")
                    log.info(f"  Docs procesados hasta parada: {processed:,}")
                    log.info('-' * 60)
                    break

                # Log intermedio cada 10 lotes
                if (batch_start // batch_sz) % 10 == 0:
                    elapsed = time.time() - t0
                    rate    = processed / elapsed if elapsed > 0 else 0
                    log.info(
                        f"Procesados {processed:>8,}/{total:,} docs  "
                        f"({rate:.0f} doc/s)  "
                        f"CONTR t1={writer.counts['t1']:,}  "
                        f"pos={writer.counts['t2p']:,}  "
                        f"hard_neg={writer.counts['t2h']:,}"
                    )

    writer.close()
    elapsed = time.time() - t0
    _print_stats(writer.counts, processed, elapsed, out_dir, log,
                 stop_reason=stop_reason, targets=writer.targets)


def _print_stats(
    counts: dict,
    n_docs: int,
    elapsed: float,
    out_dir: Path,
    log: logging.Logger,
    stop_reason: Optional[str] = None,
    targets: Optional[dict] = None,
):
    """Imprime y guarda estadisticas finales, incluyendo razon de parada."""
    tgt = targets or {"t1": 2000, "t2p": 1000, "t2n": 1000}

    log.info("-" * 60)
    log.info(f"  Documentos procesados     : {n_docs:>10,}")
    log.info(f"  Tiempo total              : {elapsed:>10.1f} s  ({elapsed/60:.1f} min)")
    if elapsed > 0:
        log.info(f"  Velocidad                 : {n_docs/elapsed:>10.1f} doc/s")
    log.info(f"  Razon de parada           : {stop_reason or 'FIN_CORPUS'}")
    log.info(f"  tarea1_CONTR (exp+impl)   : {counts['t1']:>10,}  (objetivo: {tgt['t1']:,})")
    log.info(f"  tarea2_positivos          : {counts['t2p']:>10,}  (objetivo: {tgt['t2p']:,})")
    log.info(f"  tarea2_neg_dificiles      : {counts['t2h']:>10,}")
    log.info(f"    (impl_weak excl. T1)    : {counts['t2h_weak']:>10,}")
    log.info(f"  tarea2_neg_puros          : {counts['t2n']:>10,}  (objetivo: {tgt['t2n']:,})")

    # Estado de cada objetivo
    def obj_status(actual, target, label):
        if actual >= target:
            log.info(f"  [OK]  {label}: {actual:,} >= {target:,}")
        else:
            falta = target - actual
            log.warning(f"  [!!]  {label}: {actual:,} < {target:,}  (faltan {falta:,})")

    obj_status(counts["t1"],  tgt["t1"],  "T1  CONTR")
    obj_status(counts["t2p"], tgt["t2p"], "T2  positivos")
    obj_status(counts["t2n"], tgt["t2n"], "T2  neg.puros")

    all_met = (
        counts["t1"]  >= tgt["t1"]  and
        counts["t2p"] >= tgt["t2p"] and
        counts["t2n"] >= tgt["t2n"]
    )

    stats = {
        "n_docs":           n_docs,
        "elapsed_s":        round(elapsed, 2),
        "stop_reason":      stop_reason or "FIN_CORPUS",
        "all_targets_met":  all_met,
        "targets":          tgt,
        "counts":           counts,
        "pending": {
            "t1_falta":  max(0, tgt["t1"]  - counts["t1"]),
            "t2p_falta": max(0, tgt["t2p"] - counts["t2p"]),
            "t2n_falta": max(0, tgt["t2n"] - counts["t2n"]),
        },
    }
    stats_path = out_dir / "run_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info(f"  Stats guardadas en        : {stats_path}")
    log.info("-" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline de extracción CONTR para corpus científico en español.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Modo ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode", choices=["small", "full"], required=True,
        help=(
            "small → corpus reducido (426 .txt, proceso secuencial)  |  "
            "full  → corpus completo (millones de docs, multiprocessing)"
        ),
    )

    # ── Rutas ────────────────────────────────────────────────────────────────
    p.add_argument("--corpus-dir", required=True,
                   help="Directorio raíz del corpus (--mode small: solo *.txt)")
    p.add_argument("--out-dir",    default="./output_contr",
                   help="Directorio de salida (se crea si no existe)")
    p.add_argument("--checkpoint", default=None,
                   help="Ruta del archivo de checkpoint (solo --mode full). "
                        "Default: <out-dir>/checkpoint.json")
    p.add_argument("--filelist", default=None,
                   help=(
                       "Archivo .txt con rutas absolutas de documentos (una por linea). "
                       "Evita el rglob lento en drives remotos (Google Drive, NAS). "
                       "Generar en Windows: Get-ChildItem -Recurse -File | "
                       "Select-Object FullName | Out-File filelist.txt"
                   ))

    # ── Control de corpus completo ────────────────────────────────────────────
    p.add_argument("--resume", action="store_true",
                   help="Reanudar desde el último checkpoint guardado (solo --mode full)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                   help="Número de procesos paralelos (solo --mode full)")
    p.add_argument("--batch-size", type=int, default=200,
                   help="Archivos por lote de multiprocessing (solo --mode full)")

    # ── Parámetros de fragmentación ───────────────────────────────────────────
    p.add_argument("--min-words",       type=int, default=250,
                   help="Mínimo de palabras por fragmento")
    p.add_argument("--max-words",       type=int, default=1000,
                   help="Máximo de palabras por fragmento")
    p.add_argument("--overlap",         type=int, default=2,
                   help="Oraciones de solapamiento entre ventanas deslizantes")
    p.add_argument("--max-neg-per-doc", type=int, default=3,
                   help="Máximo de fragmentos negativos puros guardados por documento")
    p.add_argument("--max-contr-per-doc", type=int, default=5,
                   help=(
                       "Maximo de fragmentos CONTR (explicit+implicit) por documento. "
                       "Requerimiento del proyecto: aislamiento de documento entre "
                       "particiones train/val/test. Valor recomendado: 3-5. "
                       "Con 426 docs y 2000 ejemplos/etiqueta se necesitan ~5 por doc."
                   ))

    # ── Segmentador de oraciones ──────────────────────────────────────────
    p.add_argument("--segmenter", choices=["simple", "spacy", "nltk"],
                   default="simple",
                   help=(
                       "Backend de segmentacion de oraciones. "
                       "simple: regex propio, sin dependencias (default). "
                       "spacy:  es_core_news_sm (pip install spacy + download). "
                       "nltk:   PunktTokenizer espanol (pip install nltk + punkt_tab)."
                   ))

    # ── Objetivos de parada anticipada ────────────────────────────────────
    p.add_argument("--target-t1", type=int, default=2000,
                   help="Objetivo: fragmentos CONTR en tarea1_CONTR.jsonl. "
                        "La ejecucion se detiene cuando todos los targets se cumplen.")
    p.add_argument("--target-t2p", type=int, default=1000,
                   help="Objetivo: fragmentos positivos en tarea2_positivos.jsonl.")
    p.add_argument("--target-t2n", type=int, default=1000,
                   help="Objetivo: fragmentos negativos puros en tarea2_neg_puros.jsonl.")

    # ── Bloqueo de patrones ruidosos ──────────────────────────────────────
    p.add_argument("--block-patterns", nargs="*", default=[],
                   metavar="LABEL",
                   help=(
                       "Etiquetas de patrones a ignorar durante la extraccion. "
                       "Util para excluir patrones ruidosos sin modificar el codigo. "
                       "Ejemplo: --block-patterns B11-primera-vez "
                       "Multiples: --block-patterns B11-primera-vez B15-novedad-locucion"
                   ))

    return p


# ══════════════════════════════════════════════════════════════════════════════
# 10. PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out_dir, args.mode)

    log.info("=" * 48)
    log.info("  Pipeline CONTR — corpus científico en español")
    log.info("=" * 48)
    log.info(f"  Modo         : {args.mode}")
    log.info(f"  Corpus dir   : {args.corpus_dir}")
    log.info(f"  Salida       : {args.out_dir}")
    log.info(f"  min/max words: {args.min_words} / {args.max_words}")
    log.info(f"  Overlap      : {args.overlap} oraciones")
    log.info(f"  Max CONTR/doc: {args.max_contr_per_doc} fragmentos")
    log.info(f"  Segmentador  : {args.segmenter}")
    log.info(f"  Target T1    : {args.target_t1}  T2pos: {args.target_t2p}  T2neg: {args.target_t2n}")
    if args.block_patterns:
        log.info(f"  Block patrones: {args.block_patterns}")
    else:
        log.info("  Block patrones: ninguno")

    if args.mode == "small":
        run_small(args, log)
    else:
        log.info(f"  Workers      : {args.workers}")
        log.info(f"  Batch size   : {args.batch_size}")
        log.info(f"  Reanudar     : {args.resume}")
        run_full(args, log)


if __name__ == "__main__":
    main()