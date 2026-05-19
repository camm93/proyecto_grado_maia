"""
T1 Preprocessing — Reproduce el pipeline de inferencia V5 byte a byte.

V5 entrenó con `neutralize_snippets(use_soft_mask=False)`, lo que significa:
- Cada snippet se enmascara con SUS PROPIOS matched_keywords (no un listado fijo).
- Los matched_keywords vienen del regex extractor del notebook AutoLabeling.

En inferencia (texto crudo nuevo), no hay matched_keywords pre-computados, así que:
1. Aplicamos el mismo regex extractor para detectar la categoría más probable y sus keywords.
2. Enmascaramos esas keywords idéntico a `neutralize_snippets` en V5.

Esto reproduce las condiciones exactas bajo las que se reportó F1 macro = 0.40 sobre gold.

NOTA: Los regex aquí son BYTE-IDÉNTICOS al notebook AutoLabeling.ipynb. Verificado.
Cualquier modificación rompe la consistencia con cómo el modelo fue entrenado.
"""

from __future__ import annotations
import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns + thresholds — IDÉNTICOS a AutoLabeling.ipynb
# ──────────────────────────────────────────────────────────────────────────────

THRESHOLDS: dict[str, int] = {
    "DISC": 2, "CONC": 2, "INTRO": 3, "LIM": 2,
    "BACK": 3, "METH": 4, "RES": 4,
}
DEFAULT_THRESHOLD = 2

REGEX_PATTERNS: dict[str, re.Pattern[str]] = {
    "INTRO": re.compile(
        r'\b('
        r'objetivo\sgeneral|objetivo\sprincipal|el\sobjetivo\sde\s(?:este|esta)|'
        r'problema\sde\sinvestigaci[oó]n|introducci[oó]n|justificaci[oó]n|'
        r'prop[oó]sito\sprincipal|planteamiento\sdel\sproblema|'
        r'el\spresente\strabajo\s(?:aborda|tiene\scomo|se\scentra)|'
        r'la\sfinalidad\sde\s(?:este|esta)|hip[oó]tesis\sde\strabajo'
        r')\b',
        re.IGNORECASE),
    "BACK": re.compile(
        r'\b(antecedentes|estado\sdel\sarte|trabajos\sprevios|'
        r'marco\ste[oó]rico|estudios\sprevios|investigaciones\sprevias|'
        r'revisi[oó]n\sbibliogr[aá]fica|literatura\scient[ií]fica|'
        r'trabajo\srelacionado|et\sal\.|seg[uú]n\s[A-Z][a-z]+|'
        r'estudios\santeriores|tradicionalmente)\b',
        re.IGNORECASE),
    "METH": re.compile(
        r'\b(metodolog[íi]a|dise[ñn]o\smetodol[oó]gico|dise[ñn]o\sde\sinvestigaci[oó]n|'
        r'materiales\sy\sm[eé]todos|material\sy\sm[eé]todo|'
        r'poblaci[oó]n\sy\smuestra|tama[ñn]o\sde\s(?:la\s)?muestra|'
        r'criterios\sde\sinclusi[oó]n|criterios\sde\sexclusi[oó]n|'
        r'instrumentos\sde\srecolecci[oó]n|recolecci[oó]n\sde\sdatos|'
        r'an[aá]lisis\sestad[íi]stico|procedimiento\sexperimental|'
        r'enfoque\scuantitativo|enfoque\scualitativo|'
        r'investigaci[oó]n\s(?:cualitativa|cuantitativa)|'
        r'entrevistas\ssemiestructuradas|aplicaci[oó]n\sde\scuestionarios|'
        r'variables\sdependientes|variables\sindependientes)\b',
        re.IGNORECASE),
    "RES": re.compile(
        r'\b('
        r'resultados|hallazgos|los\sresultados\smuestran|los\sresultados\sindican|'
        r'los\sresultados\sobtenidos|los\sdatos\smuestran|los\sanálisis\smuestran|'
        r'el\sanálisis\sindica|el\sanálisis\sreveló|'
        r'se\sencontró\sque|se\sobservó\sque|se\sidentificó|se\sdetectó|'
        r'se\sevidenció|se\sregistró|se\sobtuvieron|se\spudo\sobservar|'
        r'se\sapreció|arrojó|reveló|mostró|evidenció|'
        r'como\sse\smuestra\sen|como\sse\sobserva\sen|como\sse\saprecia\sen|'
        r'en\sla\sfigura|en\sel\sgráfico|en\sla\stabla|en\sel\scuadro|'
        r'tal\scomo\sse\smuestra|véase\sla\sfigura|véase\sla\stabla|'
        r'estadísticamente\ssignificativo|diferencia\ssignificativa|'
        r'no\ssignificativo|correlación\spositiva|correlación\snegativa|'
        r'valor\sde\sp|p\s[<>≤≥]\s0\.|intervalo\sde\sconfianza|'
        r'desviación\sestándar|error\sestándar|media\sde|mediana\sde|'
        r'coeficiente\sde|odds\sratio|razón\sde|frecuencia\srelativa|'
        r'un\saumento\sde|una\sreducción\sde|una\sdisminución\sde|'
        r'superior\sa|inferior\sa|mayor\sque|menor\sque|'
        r'el\s\d+\s?%\sde|un\s\d+\s?%|aproximadamente\sel|'
        r'los\sresultados\sobtuvieron|los\sparticipantes\sreportaron'
        r')\b',
        re.IGNORECASE),
    "DISC": re.compile(
        r'\b('
        r'discusi[oó]n|interpretaci[oó]n\sde\slos\sresultados|an[aá]lisis\scr[ií]tico|'
        r'en\sconcordancia\scon|coincide\scon|consistente\scon|'
        r'similar\sa\slo\sreportado|en\sl[ií]nea\scon|est[aá]\sde\sacuerdo\scon|'
        r'a\sdiferencia\sde|en\scontraste\scon|difiere\sde|contrario\sa|'
        r'mientras\sque\sotros\sautores|comparado\scon\sel\sestudio\sde|'
        r'estos\sresultados\ssugieren|estos\shallazgos|'
        r'esto\simplica\sque|esto\sindica\sque|lo\santerior\ssugiere|'
        r'apoya\sla\ship[oó]tesis|valida\snuestro\senfoque|corrobora\sque|'
        r'permite\sinferir|se\sinterpreta\scomo|da\scuenta\sde|'
        r'esto\sse\sdebe\sa|podr[ií]a\sexplicarse\spor|atribuible\sa|'
        r'es\sposible\sque|podr[ií]a\sdeberse\sa|una\sposible\sexplicaci[oó]n|'
        r'probablemente\sdebido\sa|factores\sque\sinfluyeron|'
        r'a\sla\sluz\sde|bajo\sesta\sperspectiva|nos\slleva\sa\spensar|'
        r'relevancia\spara|impacto\sen|consecuencia\sde|'
        r'abre\snuevas\sperspectivas|demuestra\sla\simportancia\sde'
        r')\b',
        re.IGNORECASE),
    "LIM": re.compile(
        r'\b(limitaci[oó]n|limitaciones|no\sfue\sposible|no\ses\sposible|'
        r'sesgos\spotenciales|no\sse\spuede|sesgo|sesgado|sesgada|se\sdesconoce|imposibilidad|'
        r'no\spermite|no\spermite\sgeneralizar|fuera\sdel\salcance|no\sse\spudo|no\sse\spretende|'
        r'restricci[oó]n|restringe\sla|no\sgeneraliz|no\srepresenta|tama[ñn]o\sde\sla\smuestra|'
        r'tama[ñn]o\smuestral|muestra\speque[ñn]a|futuras\sinvestigaciones|'
        r'futuras\sl[íi]neas|l[íi]neas\sfuturas|investigaci[oó]n\sfutura|'
        r'trabajo\sfuturo|queda\spendiente|falta\sde|carencia\sde|'
        r'no\sse\sdispone|pese\sa\sno\scontar|sin\scontar\scon|'
        r'no\sabarca|no\sincluye|no\scontempla|ciertas\slimitaciones|'
        r'estas\slimitaciones|dificulta|dificultades\sy\slimitaciones|debe\sinterpretarse\scon\scautela|'
        r'dise[ñn]o\stransversal\sno\spermite)\b',
        re.IGNORECASE),
    "CONC": re.compile(
        r'\b('
        r'conclusiones|conclusi[oó]n|en\sconclusi[oó]n|a\smodo\sde\sconclusi[oó]n|'
        r'para\sconcluir|como\sconclusi[oó]n|a\st[íi]tulo\sde\sconclusi[oó]n|'
        r'se\sconcluye\sque|podemos\sconcluir|cabe\sconcluir|'
        r'es\sposible\sconcluir|permite\sconcluir|lleva\sa\sconcluir|'
        r'los\sresultados\spermiten\sconcluir|los\shallazgos\spermiten\sconcluir|'
        r'en\sresumen|en\ss[íi]ntesis|en\sdefinitiva|en\s[uú]ltima\sinstancia|'
        r'implicaciones\spr[aá]cticas|como\sreflexi[oó]n\sfinal|'
        r'en\sconjunto[,\s]|tomados\sen\sconjunto|globalmente[,\s]|'
        r'en\sglobal|en\st[ée]rminos\sgenerales[,\s]|'
        r'este\sestudio\sha\sdemostrado|este\strabajo\sha\sdemostrado|'
        r'esta\sinvestigaci[oó]n\sha\sconfirmado|este\sestudio\sconfirma|'
        r'este\strabajo\saporta\sevidencia|se\sha\sdemostrado\sque|'
        r'se\sha\scomprobado\sque|se\sha\sevidenciado\sque|'
        r'los\sresultados\sobtenidos\sconfirman|'
        r'se\srecomienda|se\ssugiere\sque|futuras\sinvestigaciones\sdeber[íi]a|'
        r'pr[oó]ximas\sinvestigaciones|estudios\sfuturos\sdeber[íi]a|'
        r'es\snecesario\scontinua|queda\spendiente\spara\sfuturas|'
        r'finalmente[,\s]'
        r')\b',
        re.IGNORECASE),
}


# ──────────────────────────────────────────────────────────────────────────────
# Extractor de matched_keywords (idéntico a classify_snippet_detailed de V5)
# ──────────────────────────────────────────────────────────────────────────────

def compute_matched_keywords(snippet: str) -> tuple[Optional[str], int, list[str]]:
    """
    Replica `classify_snippet_detailed` del notebook AutoLabeling.

    Devuelve (best_category, score, matched_keywords).

    Si ninguna categoría supera su threshold, devuelve (None, 0, []).
    En ese caso, en inferencia, se devuelve el snippet sin enmascarar.
    """
    scores: dict[str, int] = {}
    matches_dict: dict[str, list[str]] = {}

    for cat, pat in REGEX_PATTERNS.items():
        matches = pat.findall(snippet)
        min_required = THRESHOLDS.get(cat, DEFAULT_THRESHOLD)
        if len(matches) >= min_required:
            scores[cat] = len(matches)
            # findall sobre grupos devuelve tuplas; aplanar
            clean = [
                (m.lower() if isinstance(m, str) else m[0].lower())
                for m in matches
            ]
            matches_dict[cat] = list(set(clean))

    if scores:
        best = max(scores, key=scores.get)
        return best, scores[best], matches_dict[best]

    return None, 0, []


# ──────────────────────────────────────────────────────────────────────────────
# Máscara dinámica (idéntica a neutralize_snippets de V5 con use_soft_mask=False)
# ──────────────────────────────────────────────────────────────────────────────

MASK_TOKEN = "[MASK]"


def apply_dynamic_mask(snippet: str, matched_keywords: list[str]) -> str:
    """
    Replica `neutralize_snippets(use_soft_mask=False)` byte a byte.

    Reemplaza las matched_keywords con MASK_TOKEN, usando word boundaries
    para no romper palabras parciales. Ordena por longitud descendente
    para evitar reemplazos prematuros (ej: "problema" antes que
    "problema de investigación").

    Si matched_keywords está vacío, devuelve el snippet original (limpio
    de espacios duplicados, como hace V5).
    """
    if not matched_keywords:
        # Mismo comportamiento que V5 cuando no hay keywords: snippet sin tocar
        return re.sub(r'\s+', ' ', str(snippet)).strip()

    out = str(snippet)
    # Ordenar por longitud DESCENDENTE: crítico para no reemplazar
    # "problema" antes que "problema de investigación"
    sorted_kws = sorted(matched_keywords, key=len, reverse=True)
    for kw in sorted_kws:
        escaped = re.escape(kw)
        pattern = re.compile(rf'\b{escaped}\b', re.IGNORECASE)
        out = pattern.sub(MASK_TOKEN, out)
    # Limpiar espacios duplicados generados por los reemplazos
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def preprocess_for_t1(snippet: str) -> dict:
    """
    Pipeline de pre-procesamiento completo para T1 SciBETO V5.

    Devuelve dict con:
        - neutralized_snippet: texto listo para tokenizar
        - matched_keywords: lista de keywords encontrados (para debug)
        - regex_category: categoría tentativa según el regex extractor
        - regex_score: número de matches de esa categoría
        - was_masked: bool, si se enmascaró algo o no
    """
    cat, score, kws = compute_matched_keywords(snippet)
    neutralized = apply_dynamic_mask(snippet, kws)
    return {
        "neutralized_snippet": neutralized,
        "matched_keywords": kws,
        "regex_category": cat,
        "regex_score": score,
        "was_masked": bool(kws),
    }
