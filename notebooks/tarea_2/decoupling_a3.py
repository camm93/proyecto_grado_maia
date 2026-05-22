#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoupling_a3.py — Actividad A3 Tarea 2
==============================================================================
Genera DOS versiones del dataset de Tarea 2 según A1 del proyecto:

  Dataset A1 (balanceado 1000/1000):
    · ES_CONTRIBUCION     → fragmentos positivos con contribución explícita
    · NO_ES_CONTRIBUCION  → fragmentos negativos puros (250-500 palabras)
    Los neg_dificiles (hard_negatives) NO forman parte del dataset A1.

  VERSIÓN "ANTES"  (dataset_a3/antes/)
    Dataset original con marcadores intactos.
    Particionado train/val (80/20) con aislamiento de documento.
    SIN test: el test de Tarea 2 es la validación humana binaria.

  VERSIÓN "DESPUÉS" (dataset_a3/despues/)
    Positivos con marcadores NEUTRALIZADOS (--neut-fraction).
    Negativos con frases CONTR INYECTADAS (--inj-fraction).
    Mismo doc→split mapping que "antes".

Salidas:
  dataset_a3/
  ├── antes/
  │   ├── train.jsonl      (~80% de 2000 = ~1600 frags)
  │   ├── val.jsonl        (~20% de 2000 = ~400 frags)
  │   └── a3_stats.json
  └── despues/
      ├── train.jsonl      (misma estructura, desacoplada)
      └── val.jsonl

TEST/EVAL: generado por select_for_annotation_t2.py (validación humana binaria)

Uso:
  python decoupling_a3.py \
      --positivos  tarea2_positivos.jsonl \
      --neg-puros  tarea2_neg_puros.jsonl \
      --out-dir    ./dataset_a3 \
      --n-pos      1000 \
      --n-neg      1000 \
      --neut-fraction 0.40 \
      --inj-fraction  0.30 \
      --train 0.80 --val 0.20 \
      --seed 42
==============================================================================
"""

import re, sys, json, random, argparse, time
from pathlib import Path
from collections import defaultdict, Counter


# ══════════════════════════════════════════════════════════════════════════════
# 1. TABLA DE NEUTRALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

NEUTRALIZATIONS: dict[str, str] = {
    "B01-demostrativo+verbo": "[TEXTO] analiza",
    "B02-1PL":                "se describe",
    "B03-pasiva+objeto":      "se utiliza un",
    "B04-presente+verbo":     "el documento incluye",
    "B05-nominal":            "los aspectos",
    "B06-principal-aport":    "el tema central",
    "B07-posesivo+objeto":    "el enfoque revisado",
    "B08-supera-SOTA":        "se compara con trabajos anteriores",
    "B09-mejores-resultados": "obtiene resultados",
    "B10-experimentos-demuestran": "los datos analizados muestran que",
    "B11-primera-vez":        "en el periodo estudiado",
    "B12-objetivo-principal": "el alcance del documento es",
    "B13-disposicion":        "se describen los recursos",
    "B14-recurso":            "se utiliza un",
    "B15-novedad-locucion":   "de forma sistemática",
    "B16-introducimos":       "se describe una",
    "B17-diferencia":         "en relación con trabajos previos",
    "B18-contribuimos":       "se presenta",
    "B19-hallazgos":          "los aspectos revisados",
    "B20-propuesta":          "el enfoque descrito",
    "I01-logro-1PL":          "se observó",
    "I02-logro-pasiva":       "se obtiene",
    "I03-superioridad":       "se compara",
    "I07-contraste":          "en relación con trabajos previos",
    "I08-avance":             "representa un aspecto relevante",
    "I10-nuestro-supera":     "el enfoque descrito muestra",
}

INJECTION_PHRASES = [
    "Este trabajo propone un nuevo enfoque para el análisis de los datos presentados a continuación.",
    "Proponemos una metodología sistemática para abordar los aspectos descritos en este documento.",
    "La principal contribución de este estudio es el análisis detallado que se presenta a continuación.",
    "Se introduce un marco de referencia para la evaluación de los elementos descritos.",
    "Nuestra propuesta consiste en el análisis de los datos y su interpretación en el contexto.",
    "A diferencia de trabajos previos, en este documento se describe el siguiente procedimiento.",
    "El presente trabajo presenta los resultados del análisis llevado a cabo.",
    "Como principal aportación, se describe el siguiente conjunto de observaciones.",
    "Los resultados obtenidos demuestran que los elementos analizados siguen los patrones esperados.",
    "Se desarrolla un análisis sistemático de los datos presentados a continuación.",
    "Este estudio contribuye con la descripción detallada del proceso analizado.",
    "El objetivo principal de este documento es presentar los datos recopilados.",
    "Presentamos el siguiente análisis basado en los datos disponibles.",
    "Nuestra metodología permite analizar los datos de forma sistemática.",
    "Se construye una base de datos con los elementos identificados en el estudio.",
    "Los experimentos realizados muestran que los datos presentados son consistentes.",
    "A continuación se introduce el enfoque seguido para la descripción de los resultados.",
    "El presente estudio analiza los aspectos mencionados a continuación.",
    "Introducimos el siguiente procedimiento para el análisis de los elementos estudiados.",
    "Este trabajo aporta la descripción sistemática de los procesos observados.",
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. NEUTRALIZACIÓN E INYECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def neutralize_fragment(frag: dict) -> dict:
    """Reemplaza los matches de los patrones por texto neutral."""
    text = frag["text"]
    hits = frag.get("explicit_hits", []) + frag.get("implicit_hits", [])
    neutralized_hits = []
    for hit in hits:
        replace = NEUTRALIZATIONS.get(hit["label"], "")
        pattern = re.compile(re.escape(hit["match"]), re.IGNORECASE)
        new_text = pattern.sub(replace, text, count=1)
        if new_text != text:
            neutralized_hits.append({
                "label":    hit["label"],
                "original": hit["match"],
                "replaced": replace,
            })
            text = new_text
    result = dict(frag)
    result["text"]             = text
    result["neutralized_hits"] = neutralized_hits
    result["decoupling"]       = "neutralized" if neutralized_hits else "unchanged"
    return result


def inject_phrase(frag: dict, phrase: str) -> dict:
    """Antepone una frase CONTR al fragmento negativo."""
    result = dict(frag)
    result["text"]       = phrase + " " + frag["text"]
    result["injected"]   = phrase
    result["decoupling"] = "injected"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. SELECCIÓN BALANCEADA
# ══════════════════════════════════════════════════════════════════════════════

def select_balanced(positivos, neg_puros, n_pos, n_neg, rng):
    """
    Selecciona n_pos positivos y n_neg negativos puros.
    Negativos: solo los que están en rango 250-500 palabras (A1).
    Positivos: orden por score descendente para máxima calidad.
    """
    # Negativos: filtrar rango y samplear
    neg_in_range = [f for f in neg_puros if 250 <= f["word_count"] <= 500]
    if len(neg_in_range) < n_neg:
        print(f"  [AVISO] Solo {len(neg_in_range)} neg. en rango 250-500w "
              f"(pedidos {n_neg}). Usando todos.")
        n_neg = len(neg_in_range)
    neg_selected = rng.sample(neg_in_range, n_neg)

    # Positivos: tomar los de mayor score compuesto
    pos_sorted = sorted(
        positivos,
        key=lambda f: f.get("score_explicit", 0) + len(f.get("explicit_hits", [])) * 0.1,
        reverse=True
    )
    pos_selected = pos_sorted[:n_pos]

    return pos_selected, neg_selected


# ══════════════════════════════════════════════════════════════════════════════
# 4. PARTICIONAMIENTO TRAIN/VAL (sin test)
# ══════════════════════════════════════════════════════════════════════════════

def assign_splits(fragments, train_r, val_r, seed):
    """
    Asigna cada doc_id a train o val con aislamiento de documento.
    SIN test: el test de Tarea 2 proviene de la validación humana binaria.
    """
    rng = random.Random(seed)
    doc_frags = defaultdict(list)
    for f in fragments:
        doc_frags[f["doc_id"]].append(f)

    docs = list(doc_frags.keys())
    rng.shuffle(docs)

    total_frags  = len(fragments)
    target_train = int(total_frags * train_r)

    assignment = {}
    counts = {"train": 0, "val": 0}
    for doc_id in docs:
        n = len(doc_frags[doc_id])
        split = "train" if counts["train"] < target_train else "val"
        assignment[doc_id] = split
        counts[split] += n

    return assignment, counts


# ══════════════════════════════════════════════════════════════════════════════
# 5. ESCRITURA
# ══════════════════════════════════════════════════════════════════════════════

def write_splits(fragments, assignment, out_dir, version):
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = {
        "train": open(out_dir / "train.jsonl", "w", encoding="utf-8"),
        "val":   open(out_dir / "val.jsonl",   "w", encoding="utf-8"),
    }
    counts = {"train": 0, "val": 0}
    label_per_split = defaultdict(Counter)

    for frag in fragments:
        split = assignment.get(frag["doc_id"], "train")
        out   = dict(frag)
        out["split"]   = split
        out["version"] = version
        fh[split].write(json.dumps(out, ensure_ascii=False) + "\n")
        counts[split] += 1
        label_per_split[split][frag.get("a1_label", "?")] += 1

    for f in fh.values():
        f.flush(); f.close()

    return {"counts": counts, "labels": {s: dict(l) for s, l in label_per_split.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        print(f"  [AVISO] No encontrado: {path}"); return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def run(args):
    rng     = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0      = time.time()

    print("=" * 65)
    print("  A3 Tarea 2 — Desacoplamiento de patrones")
    print("=" * 65)
    print(f"  Dataset A1       : {args.n_pos} positivos + {args.n_neg} negativos puros")
    print(f"  neut_fraction    : {args.neut_fraction:.0%}  (positivos a neutralizar)")
    print(f"  inj_fraction     : {args.inj_fraction:.0%}   (negativos con frase inyectada)")
    print(f"  split            : {args.train:.0%}/{args.val:.0%}  (train/val)")
    print(f"  test             : validacion humana binaria (externo)")
    print(f"  seed             : {args.seed}")
    print()

    # Cargar
    positivos = load_jsonl(args.positivos)
    neg_puros = load_jsonl(args.neg_puros)
    print(f"Cargados: positivos={len(positivos):,}  neg_puros={len(neg_puros):,}")

    # Selección balanceada
    pos_sel, neg_sel = select_balanced(
        positivos, neg_puros, args.n_pos, args.n_neg, rng
    )
    print(f"Seleccionados: {len(pos_sel)} positivos + {len(neg_sel)} negativos = "
          f"{len(pos_sel)+len(neg_sel)} total")

    # Añadir etiqueta A1 binaria
    for f in pos_sel:
        f["a1_label"] = "ES_CONTRIBUCION"
        f["a1_class"] = 1
    for f in neg_sel:
        f["a1_label"] = "NO_ES_CONTRIBUCION"
        f["a1_class"] = 0

    all_frags = pos_sel + neg_sel

    # Asignar splits
    print("\nAsignando splits con aislamiento de documento...")
    assignment, counts_approx = assign_splits(all_frags, args.train, args.val, args.seed)
    n_train_docs = sum(1 for s in assignment.values() if s == "train")
    n_val_docs   = sum(1 for s in assignment.values() if s == "val")
    print(f"  Docs asignados: train={n_train_docs}  val={n_val_docs}")
    print(f"  (TEST = validación humana binaria, externo)")

    # ── VERSIÓN ANTES ─────────────────────────────────────────────────────────
    print("\nGenerando VERSIÓN ANTES (marcadores intactos)...")
    stats_antes = write_splits(all_frags, assignment, out_dir / "antes", "antes")
    t, v = stats_antes["counts"]["train"], stats_antes["counts"]["val"]
    print(f"  train={t:,}  val={v:,}")
    for split, labels in stats_antes["labels"].items():
        print(f"    {split}: {labels}")

    # ── VERSIÓN DESPUÉS ───────────────────────────────────────────────────────
    print("\nGenerando VERSIÓN DESPUÉS (desacoplada)...")

    # Neutralizar fracción de positivos
    n_neut = int(len(pos_sel) * args.neut_fraction)
    idx_neut = set(rng.sample(range(len(pos_sel)), n_neut))
    pos_despues = []
    n_changed = 0
    for i, f in enumerate(pos_sel):
        fn = neutralize_fragment(f) if i in idx_neut else dict(f)
        if fn.get("decoupling") == "neutralized":
            n_changed += 1
        elif "decoupling" not in fn:
            fn["decoupling"] = "intact"
        pos_despues.append(fn)

    # Inyectar frases en fracción de negativos
    n_inj = int(len(neg_sel) * args.inj_fraction)
    idx_inj = set(rng.sample(range(len(neg_sel)), n_inj))
    neg_despues = []
    for i, f in enumerate(neg_sel):
        if i in idx_inj:
            fn = inject_phrase(f, rng.choice(INJECTION_PHRASES))
        else:
            fn = dict(f)
            fn["decoupling"] = "intact"
        neg_despues.append(fn)

    print(f"  Positivos neutralizados: {n_changed}/{n_neut}  "
          f"(de {len(pos_sel)} totales)")
    print(f"  Negativos inyectados   : {n_inj}/{len(neg_sel)}")

    all_despues = pos_despues + neg_despues
    stats_despues = write_splits(all_despues, assignment, out_dir / "despues", "despues")
    t2, v2 = stats_despues["counts"]["train"], stats_despues["counts"]["val"]
    print(f"  train={t2:,}  val={v2:,}")
    for split, labels in stats_despues["labels"].items():
        print(f"    {split}: {labels}")

    # ── VERIFICACIÓN DE AISLAMIENTO ───────────────────────────────────────────
    doc_splits = defaultdict(set)
    for frag in all_frags:
        doc_splits[frag["doc_id"]].add(assignment.get(frag["doc_id"], "train"))
    violations = [d for d, s in doc_splits.items() if len(s) > 1]

    # ── REPORTE FINAL ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 65)
    print("  REPORTE FINAL A3")
    print("=" * 65)
    print(f"  Tiempo                 : {elapsed:.1f}s")
    print(f"  Dataset A1 total       : {len(all_frags):,}  "
          f"(ES_CONTR={len(pos_sel):,} | NO_ES_CONTR={len(neg_sel):,})")
    print(f"  Versión ANTES          : train={t:,}  val={v:,}")
    print(f"  Versión DESPUÉS        : train={t2:,}  val={v2:,}")
    print(f"  Neutralizados          : {n_changed}/{len(pos_sel)}")
    print(f"  Inyectados             : {n_inj}/{len(neg_sel)}")
    if violations:
        print(f"  [ERROR] Violaciones de aislamiento: {len(violations)}")
    else:
        print(f"  [OK] Aislamiento de documento garantizado")
    print()
    print("  Archivos generados:")
    for version in ["antes", "despues"]:
        for split in ["train", "val"]:
            p = out_dir / version / f"{split}.jsonl"
            if p.exists():
                n = sum(1 for _ in open(p, encoding="utf-8"))
                print(f"    {version}/{split}.jsonl : {n:,}")

    # Guardar stats
    stats = {
        "config": {
            "n_pos": args.n_pos, "n_neg": args.n_neg,
            "neut_fraction": args.neut_fraction,
            "inj_fraction":  args.inj_fraction,
            "seed": args.seed,
            "split": {"train": args.train, "val": args.val},
            "test_source": "validacion_humana_binaria",
        },
        "dataset_a1": {
            "ES_CONTRIBUCION":    len(pos_sel),
            "NO_ES_CONTRIBUCION": len(neg_sel),
            "total":              len(all_frags),
        },
        "decoupling": {
            "positivos_neutralizados": n_changed,
            "negativos_inyectados":    n_inj,
        },
        "antes":   stats_antes,
        "despues": stats_despues,
        "isolation_ok": len(violations) == 0,
        "elapsed_s": round(elapsed, 2),
    }
    (out_dir / "a3_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Stats: {out_dir / 'a3_stats.json'}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="A3 Tarea 2: desacoplamiento. Dataset A1: positivos + neg_puros.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--positivos",  required=True, help="tarea2_positivos.jsonl")
    p.add_argument("--neg-puros",  required=True, help="tarea2_neg_puros.jsonl")
    p.add_argument("--out-dir",    default="./dataset_a3")

    p.add_argument("--n-pos",  type=int, default=1000,
                   help="Positivos a incluir en el dataset A1 (≥1000 requerido).")
    p.add_argument("--n-neg",  type=int, default=1000,
                   help="Negativos puros a incluir (≥1000, rango 250-500w).")

    p.add_argument("--neut-fraction", type=float, default=0.40,
                   help="Fracción de positivos a neutralizar (0.0-1.0).")
    p.add_argument("--inj-fraction",  type=float, default=0.30,
                   help="Fracción de negativos con frase CONTR inyectada.")

    p.add_argument("--train", type=float, default=0.80)
    p.add_argument("--val",   type=float, default=0.20)
    p.add_argument("--seed",  type=int,   default=42)

    args = p.parse_args()
    if abs(args.train + args.val - 1.0) > 0.01:
        print("[ERROR] train+val debe sumar 1.0"); sys.exit(1)
    if args.n_pos < 1000 or args.n_neg < 1000:
        print("[ERROR] n_pos y n_neg deben ser ≥1000 (requisito A1)"); sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()