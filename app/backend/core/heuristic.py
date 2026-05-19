"""
Motores heurísticos puros para T1 (segmentación retórica) y T2 (contribuciones).

Estos motores son la baseline siempre disponible y el fallback cuando un modelo
fine-tuned aún no está descargado.
"""

from __future__ import annotations

import re

from .lexicon import T1_PATTERNS, T1_PRIORITY, T1_NAMES, T1_COLORS, T2_RE


# ── Segmentador de párrafos ──────────────────────────────────────────────────
def split_paragraphs(text: str) -> list[str]:
    by_double = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(by_double) > 1:
        return by_double

    by_single = [p.strip() for p in text.split('\n') if p.strip()]
    if len(by_single) > 3:
        groups, cur, wc = [], [], 0
        for line in by_single:
            lw = len(line.split())
            is_header = lw <= 5 and bool(re.match(r'^[A-ZAEIOUÜN]', line))
            if is_header and cur:
                groups.append(' '.join(cur)); cur, wc = [], 0
            cur.append(line); wc += lw
            if is_header or wc >= 25:
                groups.append(' '.join(cur)); cur, wc = [], 0
        if cur:
            groups.append(' '.join(cur))
        return [g for g in groups if g.strip()]

    sents = re.split(r'(?<=[.!?])\s+', text)
    groups, current, wc = [], [], 0
    for s in sents:
        current.append(s); wc += len(s.split())
        if wc >= 25:
            groups.append(' '.join(current)); current, wc = [], 0
    if current:
        groups.append(' '.join(current))
    return [g for g in groups if g.strip()] or [text]


# ── T1: clasificación retórica por párrafo ───────────────────────────────────
def run_t1(text: str) -> list[dict]:
    """Clasifica párrafos retóricamente. Devuelve lista de segmentos con metadata."""
    paragraphs = split_paragraphs(text)
    n = len(paragraphs)
    results: list[dict] = []
    char_offset = 0

    for idx, para in enumerate(paragraphs):
        pos = idx / max(n - 1, 1)
        scores = {lbl: len(pat.findall(para)) for lbl, pat in T1_PATTERNS.items()}
        if pos <= 0.15:
            scores["INTRO"] += 3; scores["BACK"] += 2
        elif pos >= 0.80:
            scores["CONC"] += 3; scores["LIM"] += 2

        best = max(T1_PRIORITY, key=lambda l: scores[l])
        total = sum(scores.values()) or 1
        t1c = round(min(0.95, 0.45 + scores[best] / total * 0.5), 3)
        low = scores[best] == 0
        if low:
            best = ["INTRO", "BACK", "METH", "RES", "DISC", "CONC"][min(5, int(pos * 6))]
            t1c = 0.42

        cs = text.find(para, char_offset)
        char_start = cs if cs >= 0 else char_offset
        char_end = char_start + len(para)
        char_offset = char_end

        zone = ("Inicio (0-20%)" if pos <= 0.20
                else "Cierre (80-100%)" if pos >= 0.80
                else "Desarrollo (20-80%)")

        results.append({
            "index": idx, "text": para,
            "char_start": char_start, "char_end": char_end,
            "word_count": len(para.split()),
            "relative_pos": round(pos, 4),
            "rhetorical_zone": zone,
            "t1_label": best, "t1_label_name": T1_NAMES[best],
            "t1_color": T1_COLORS[best], "t1_confidence": t1c,
            "low_confidence": low,
        })

    return results


# ── T2: detección binaria de contribuciones por fragmento ────────────────────
def run_t2(fragments: list[dict]) -> list[dict]:
    """Clasifica cada fragmento como ES/NO_ES_CONTRIBUCION.
    Recibe los fragmentos ya etiquetados por T1 (incluye contexto retórico).
    """
    results: list[dict] = []
    for frag in fragments:
        text = frag.get("text", "")
        n = len(T2_RE.findall(text))
        if n >= 3:
            is_c, conf = True, min(0.95, 0.70 + n * 0.04)
        elif n == 2:
            is_c, conf = True, 0.80
        elif n == 1:
            is_c, conf = True, 0.65
        else:
            is_c, conf = False, max(0.48, 0.85 - len(text.split()) * 0.0009)

        t1_name = frag.get("t1_label_name", frag.get("t1_label", ""))
        t1_conf = frag.get("t1_confidence", 0)
        zone = frag.get("rhetorical_zone", "")
        ctx = (f"Contribucion detectada en seccion '{t1_name}' "
               f"({t1_conf*100:.0f}% conf. retorica) | {zone}")

        results.append({
            "index": frag.get("index", 0),
            "is_contribution": is_c,
            "confidence": round(conf, 3),
            "label": "ES_CONTRIBUCION" if is_c else "NO_ES_CONTRIBUCION",
            "t1_label": frag.get("t1_label", ""),
            "t1_label_name": t1_name,
            "t1_color": frag.get("t1_color", "#9CA3AF"),
            "t1_confidence": t1_conf,
            "rhetorical_zone": zone,
            "rhetorical_context": ctx if is_c else "",
            "char_start": frag.get("char_start"),
            "char_end": frag.get("char_end"),
            "word_count": frag.get("word_count"),
            "relative_pos": frag.get("relative_pos"),
        })

    return results
