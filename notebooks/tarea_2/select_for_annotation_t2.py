#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_for_annotation_t2.py — Actividad A2 Tarea 2
==============================================================================
Selecciona el 10% del dataset A1 para VALIDACIÓN HUMANA BINARIA.

Dataset A1 (según requerimiento):
  · ES_CONTRIBUCION     → fragmentos positivos
  · NO_ES_CONTRIBUCION  → fragmentos negativos puros (250-500 palabras)

El resultado es el TEST/EVAL oficial de Tarea 2:
  100 positivos + 100 negativos = 200 fragmentos  (~10% del dataset A1)

Salidas:
  <out-dir>/anotacion_anotador1.csv   → para el Anotador 1 (label vacío)
  <out-dir>/anotacion_anotador2.csv   → para el Anotador 2 (label vacío)
  <out-dir>/anotacion_referencia.csv  → con auto_label (NO mostrar al anotador)
  <out-dir>/guia_anotacion.txt        → instrucciones para anotadores

Campos del CSV de anotación:
  id, source_file, snippet, label (vacío), comentario

Uso:
  python select_for_annotation_t2.py \
      --positivos  tarea2_positivos.jsonl \
      --neg-puros  tarea2_neg_puros.jsonl \
      --out-dir    ./anotacion_t2 \
      --n-pos      100 \
      --n-neg      100 \
      --seed       42
==============================================================================
"""

import csv, sys, json, random, argparse
from pathlib import Path
from collections import Counter


FIELDS_ANNOTATOR = [
    "id",
    "source_file",
    "snippet",
    "label",       # el anotador escribe 1 o 0
    "comentario",  # opcional
]

FIELDS_REFERENCE = [
    "id",
    "source_file",
    "auto_label",       # 1=ES_CONTRIBUCION, 0=NO_ES_CONTRIBUCION
    "a1_class_auto",    # clase asignada automáticamente
    "contr_type",
    "score_explicit",
    "key_phrases",
    "relative_pos",
    "word_count",
    "snippet",
]

GUIA = """\
GUÍA DE ANOTACIÓN — TAREA 2: Extracción de Contribuciones Científicas
=======================================================================

TAREA: Indica si el fragmento describe una CONTRIBUCIÓN CIENTÍFICA ORIGINAL.

ETIQUETA BINARIA:
  1 = ES_CONTRIBUCION
  0 = NO_ES_CONTRIBUCION

INSTRUCCIONES:
  · Lea el fragmento completo antes de asignar la etiqueta.
  · Use el campo 'comentario' para casos dudosos.
  · Trabaje de forma independiente (no consulte al otro anotador).
  · Anote todos los fragmentos aunque el caso sea difícil.

CRITERIOS PARA  1 (ES_CONTRIBUCION):
  ✓ Propone un método, sistema, arquitectura o algoritmo nuevo.
  ✓ Presenta un corpus o dataset construido por los autores.
  ✓ Demuestra que su enfoque supera a trabajos previos con evidencia.
  ✓ Describe un hallazgo empírico original no reportado antes.
  ✓ Introduce una métrica, marco conceptual o taxonomía nueva.
  ✓ Aporta evidencia empírica sobre un fenómeno no estudiado.

CRITERIOS PARA  0 (NO_ES_CONTRIBUCION):
  ✗ Describe metodología estándar sin novedad.
  ✗ Resume o parafrasea trabajos previos.
  ✗ Presenta resultados descriptivos sin aporte novedoso.
  ✗ Define conceptos del dominio sin contribución propia.
  ✗ Usa frases como "propone" pero el contenido no lo respalda.
  ✗ Discute implicaciones sin introducir novedad propia.

CASOS DIFÍCILES:
  · Si hay frases de contribución al inicio pero el cuerpo es
    metodológico sin novedad → 0
  · Mejoras incrementales cuantificadas con datos → 1
  · Descripciones de trabajo futuro sin resultados → 0

FORMATO:
  · Columna 'label': escriba SOLO  1  o  0
  · Columna 'comentario': texto libre, opcional

Tiempo estimado: ~4 min por fragmento × {n_total} fragmentos = ~{total_min} min
"""


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        print(f"  [AVISO] No encontrado: {path}"); return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def stratified_pos(frags, n, rng):
    """Muestra de positivos estratificada por zona del documento."""
    def zona(pos):
        if pos <= 0.15: return "intro"
        if pos <= 0.40: return "desarrollo"
        if pos <= 0.70: return "cuerpo"
        return "final"

    grupos = {}
    for f in frags:
        grupos.setdefault(zona(f["relative_pos"]), []).append(f)

    result = []
    total  = len(frags)
    for g in grupos.values():
        k = max(1, round(n * len(g) / total))
        result.extend(rng.sample(g, min(k, len(g))))
    rng.shuffle(result)

    # Completar si hace falta
    if len(result) < n:
        usados = set(id(f) for f in result)
        resto  = [f for f in frags if id(f) not in usados]
        rng.shuffle(resto)
        result.extend(resto[:n - len(result)])
    return result[:n]


def stratified_neg(frags, n, rng):
    """Muestra de negativos puros filtrada por rango 250-500w."""
    en_rango = [f for f in frags if 250 <= f["word_count"] <= 500]
    if len(en_rango) < n:
        print(f"  [AVISO] Solo {len(en_rango)} neg. en rango. Ajustando n_neg.")
        n = len(en_rango)
    return rng.sample(en_rango, n)


def run(args):
    rng     = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  A2 Tarea 2 — Selección para validación humana binaria")
    print("=" * 65)
    print(f"  n_pos (ES_CONTRIBUCION)    : {args.n_pos}")
    print(f"  n_neg (NO_ES_CONTRIBUCION) : {args.n_neg}")
    print(f"  Total a anotar             : {args.n_pos + args.n_neg}")
    print(f"  Seed                       : {args.seed}")
    print()

    positivos = load_jsonl(args.positivos)
    neg_puros = load_jsonl(args.neg_puros)
    print(f"Cargados: positivos={len(positivos):,}  neg_puros={len(neg_puros):,}")

    sample_pos = stratified_pos(positivos, args.n_pos, rng)
    sample_neg = stratified_neg(neg_puros,  args.n_neg, rng)
    print(f"Seleccionados: pos={len(sample_pos)}  neg={len(sample_neg)}")

    # Combinar y barajar (el anotador no ve el orden por clase)
    all_samples = (
        [(f, 1) for f in sample_pos] +
        [(f, 0) for f in sample_neg]
    )
    rng.shuffle(all_samples)

    rows_ann = []
    rows_ref = []
    for idx, (frag, auto_label) in enumerate(all_samples, start=1):
        frag_id = f"T2-{idx:04d}"
        phrases = "; ".join(
            h.get("match", "") for h in
            frag.get("explicit_hits", []) + frag.get("implicit_hits", [])
        )
        snippet = " ".join(frag.get("text", "").split())

        rows_ann.append({
            "id":          frag_id,
            "source_file": frag.get("doc_id", "") + ".txt",
            "snippet":     snippet,
            "label":       "",
            "comentario":  "",
        })
        rows_ref.append({
            "id":            frag_id,
            "source_file":   frag.get("doc_id", "") + ".txt",
            "auto_label":    auto_label,
            "a1_class_auto": "ES_CONTRIBUCION" if auto_label == 1 else "NO_ES_CONTRIBUCION",
            "contr_type":    frag.get("contr_type", ""),
            "score_explicit": frag.get("score_explicit", 0),
            "key_phrases":   phrases[:200],
            "relative_pos":  frag.get("relative_pos", 0),
            "word_count":    frag.get("word_count", 0),
            "snippet":       snippet,
        })

    def write_csv(path, fieldnames, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            w.writeheader(); w.writerows(rows)

    write_csv(out_dir / "anotacion_anotador1.csv", FIELDS_ANNOTATOR, rows_ann)
    write_csv(out_dir / "anotacion_anotador2.csv", FIELDS_ANNOTATOR, rows_ann)
    write_csv(out_dir / "anotacion_referencia.csv", FIELDS_REFERENCE, rows_ref)

    n_total   = len(all_samples)
    total_min = n_total * 4
    (out_dir / "guia_anotacion.txt").write_text(
        GUIA.format(n_total=n_total, total_min=total_min), encoding="utf-8"
    )

    # Distribución real en la muestra
    dist = Counter("ES_CONTRIBUCION" if a == 1 else "NO_ES_CONTRIBUCION"
                   for _, a in all_samples)

    print()
    print("=" * 65)
    print("  ARCHIVOS GENERADOS")
    print("=" * 65)
    print(f"  anotacion_anotador1.csv  → {args.n_pos + args.n_neg} filas (label vacío)")
    print(f"  anotacion_anotador2.csv  → copia para 2do anotador")
    print(f"  anotacion_referencia.csv → con auto_label (NO mostrar al anotador)")
    print(f"  guia_anotacion.txt")
    print()
    print("  Distribución de la muestra (orden aleatorio en CSV):")
    for cls, cnt in dist.items():
        print(f"    {cls:<25}: {cnt}")
    print(f"  Tiempo estimado/anotador : ~{total_min} min (~{total_min//60}h {total_min%60}min)")
    print()
    print("  INSTRUCCIONES COORDINADOR:")
    print("  1. Entregar anotacion_anotador1.csv → Anotador 1")
    print("  2. Entregar anotacion_anotador2.csv → Anotador 2")
    print("  3. No compartir entre anotadores hasta que ambos terminen")
    print("  4. Tras la anotación ejecutar:")
    print("     python compute_agreement_t2.py \\")
    print("         --ann1 anotacion_anotador1.csv \\")
    print("         --ann2 anotacion_anotador2.csv \\")
    print("         --ref  anotacion_referencia.csv \\")
    print("         --out  acuerdo_interanotador.json")
    print("=" * 65)


def main():
    p = argparse.ArgumentParser(
        description="A2 Tarea 2: selección 10%% para validación humana binaria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--positivos", required=True, help="tarea2_positivos.jsonl")
    p.add_argument("--neg-puros", required=True, help="tarea2_neg_puros.jsonl")
    p.add_argument("--out-dir",   default="./anotacion_t2")
    p.add_argument("--n-pos", type=int, default=100,
                   help="Positivos a seleccionar (recomendado: 10%% de los disponibles).")
    p.add_argument("--n-neg", type=int, default=100,
                   help="Negativos puros a seleccionar.")
    p.add_argument("--seed",  type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = main()
    run(args)