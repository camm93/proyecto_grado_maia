"""
Lexicón compartido entre backend y frontend.

Las cadenas de regex viven aquí como única fuente de verdad. El backend las
compila con `re.I`; el frontend las consume vía GET /api/lexicon y las
compila con la flag `gi`.
"""

from __future__ import annotations

import re

# ── T1: patrones por etiqueta retórica ───────────────────────────────────────
T1_PATTERN_SOURCES: dict[str, str] = {
    "INTRO": r'\b(objetivo|objetivos|introducci[oó]n|problema\sde\sinvestigaci[oó]n|justificaci[oó]n|prop[oó]sito|el\spresente\strabajo|esta\sinvestigaci[oó]n|planteamiento)\b',
    "BACK":  r'\b(antecedentes|estado\sdel\sarte|trabajos\sprevios|marco\ste[oó]rico|estudios\sprevios|literatura|seg[uú]n|et\sal|autores\scomo)\b',
    "METH":  r'\b(metodolog[íi]a|m[eé]todo|procedimiento|dise[nñ]o|muestra|participantes|cuestionario|an[aá]lisis\sestad[íi]stico|tipo\sde\sestudio|muestreo)\b',
    "RES":   r'\b(resultados|hallazgos|los\sresultados|se\sencontr[oó]|se\sobserv[oó]|estad[íi]sticamente|correlaci[oó]n|se\sobtuvieron|en\sla\stabla|valor\sde\sp)\b',
    "DISC":  r'\b(discusi[oó]n|estos\sresultados\ssugieren|esto\simplica|estos\shallazgos|en\scontraste\scon|coincide\scon|difiere\sde|lo\scual\sindica)\b',
    "CONTR": r'\b(contribuci[oó]n|nuestro\senfoque|proponemos|presentamos|introducimos|nuestra\spropuesta|a\sdiferencia\sde\strabajos|se\spropone\sun\snuevo|supera\sal\sestado|la\saportaci[oó]n|aportaci[oó]n)\b',
    "LIM":   r'\b(limitaci[oó]n|limitaciones|no\sfue\sposible|sesgo|tama[nñ]o\smuestral|muestra\speque[nñ]a|futuras\sinvestigaciones|trabajo\sfuturo|restricci[oó]n)\b',
    "CONC":  r'\b(conclusi[oó]n|conclusiones|en\sconclusi[oó]n|para\sconcluir|se\sconcluye\sque|en\sresumen|finalmente|se\srecomienda)\b',
}

T1_PATTERNS: dict[str, re.Pattern[str]] = {
    label: re.compile(src, re.I) for label, src in T1_PATTERN_SOURCES.items()
}

T1_PRIORITY: list[str] = ["CONTR", "INTRO", "CONC", "LIM", "METH", "BACK", "RES", "DISC"]

T1_COLORS: dict[str, str] = {
    "INTRO": "#2563EB", "BACK": "#7C3AED", "METH": "#84CC16", "RES": "#16A34A",
    "DISC": "#0F766E", "CONTR": "#DC2626", "LIM": "#C026D3", "CONC": "#D97706",
}

T1_NAMES: dict[str, str] = {
    "INTRO": "Introducción", "BACK": "Antecedentes", "METH": "Metodología",
    "RES": "Resultados", "DISC": "Discusión", "CONTR": "Contribución",
    "LIM": "Limitaciones", "CONC": "Conclusión",
}

# ── T2: patrón único para detección de contribuciones ────────────────────────
T2_PATTERN_SOURCE: str = (
    r'\b(proponemos|presentamos|introducimos|desarrollamos|contribuimos|aportamos|'
    r'se\spropone|se\spresenta|se\sintroduce|se\sdesarrolla|se\sconstruye|se\screa|'
    r'este\strabajo\sporta|nuestra\spropuesta|nuestro\sm[eé]todo|nuestro\senfoque|'
    r'nuestra\sarquitectura|nuestro\smodelo|nuestra\ssoluci[oó]n|'
    r'supera\sal\sestado\sdel\sarte|a\sdiferencia\sde\strabajos|'
    r'la\sprincipal\saportaci[oó]n|la\sprincipal\scontribuci[oó]n|contribuci[oó]n|'
    r'aportaci[oó]n|el\sobjetivo\sde\seste\strabajo\ses|ponemos\sa\sdisposici[oó]n|'
    r'nuevo\sm[eé]todo|nuevo\senfoque|nueva\sarquitectura|nueva\smetodolog[íi]a)\b'
)

T2_RE: re.Pattern[str] = re.compile(T2_PATTERN_SOURCE, re.I)


def as_payload() -> dict:
    """Forma serializable del lexicón para el endpoint /api/lexicon."""
    return {
        "t1": {
            "patterns": T1_PATTERN_SOURCES,
            "priority": T1_PRIORITY,
            "colors":   T1_COLORS,
            "names":    T1_NAMES,
        },
        "t2": {"pattern": T2_PATTERN_SOURCE},
    }
