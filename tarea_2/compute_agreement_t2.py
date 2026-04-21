#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_agreement_t2.py — Acuerdo interanotador Tarea 2
==============================================================================
Calcula métricas de acuerdo interanotador para la validación humana binaria
del dataset de Tarea 2 (ES_CONTRIBUCION / NO_ES_CONTRIBUCION).

Métricas calculadas:
  · Cohen's Kappa (κ)      — acuerdo entre los 2 anotadores
  · Krippendorff's Alpha   — acuerdo generalizado (esquema binario)
  · Acuerdo simple (%)     — porcentaje de coincidencia directa
  · Matriz de confusión    — entre los dos anotadores
  · Acuerdo vs auto_label  — cada anotador vs clasificación automática
  · Casos de desacuerdo    — listado de fragmentos donde difieren

Requisito del proyecto: κ ≥ 0.60 (acuerdo sustancial)

Uso:
  python compute_agreement_t2.py \
      --ann1 anotacion_anotador1.csv \
      --ann2 anotacion_anotador2.csv \
      --ref  anotacion_referencia.csv \
      --out  acuerdo_interanotador.json

  # Ver también los casos de desacuerdo:
  python compute_agreement_t2.py \
      --ann1 anotacion_anotador1.csv \
      --ann2 anotacion_anotador2.csv \
      --ref  anotacion_referencia.csv \
      --out  acuerdo_interanotador.json \
      --show-disagreements
==============================================================================
"""

import csv
import sys
import json
import argparse
from pathlib import Path
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
# 1. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """
    Cohen's Kappa para dos anotadores y etiquetas binarias {0, 1}.

    κ = (P_o - P_e) / (1 - P_e)
      P_o = acuerdo observado
      P_e = acuerdo esperado por azar
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Acuerdo observado
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o   = agree / n

    # Distribución marginal
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    classes = set(count_a) | set(count_b)

    # Acuerdo esperado por azar
    p_e = sum((count_a.get(c, 0) / n) * (count_b.get(c, 0) / n) for c in classes)

    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)


def krippendorff_alpha(labels_a: list[int], labels_b: list[int]) -> float:
    """
    Krippendorff's Alpha para esquema binario nominal con 2 anotadores.

    α = 1 - (D_o / D_e)
      D_o = desacuerdo observado
      D_e = desacuerdo esperado
    Para escala nominal: d(c,k) = 0 si c==k, 1 si c!=k
    """
    n = len(labels_a)
    if n == 0:
        return 0.0

    # Desacuerdo observado (por par de anotaciones)
    d_o = sum(1 for a, b in zip(labels_a, labels_b) if a != b) / n

    # Distribución de valores (todos los valores de ambos anotadores)
    all_vals = labels_a + labels_b
    n_total  = len(all_vals)
    count    = Counter(all_vals)
    classes  = list(count.keys())

    # Desacuerdo esperado
    d_e = 0.0
    for c in classes:
        for k in classes:
            if c != k:
                d_e += (count[c] / n_total) * (count[k] / n_total)

    if d_e == 0.0:
        return 1.0
    return round(1.0 - (d_o / d_e), 4)


def percent_agreement(labels_a: list[int], labels_b: list[int]) -> float:
    n = len(labels_a)
    if n == 0: return 0.0
    return round(sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n * 100, 2)


def confusion_matrix_2x2(labels_a: list[int], labels_b: list[int]) -> dict:
    """
    Matriz de confusión entre dos anotadores.
    Filas = Anotador 1, Columnas = Anotador 2.
    """
    tp = sum(1 for a, b in zip(labels_a, labels_b) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(labels_a, labels_b) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(labels_a, labels_b) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(labels_a, labels_b) if a == 1 and b == 0)
    return {"A1=1_A2=1": tp, "A1=0_A2=0": tn,
            "A1=0_A2=1": fp, "A1=1_A2=0": fn}


def kappa_interpretation(kappa: float) -> str:
    if kappa < 0.00: return "Sin acuerdo"
    if kappa < 0.20: return "Leve"
    if kappa < 0.40: return "Aceptable"
    if kappa < 0.60: return "Moderado"
    if kappa < 0.80: return "Sustancial  ← mínimo requerido por el proyecto"
    return "Casi perfecto"


# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE CSVs
# ══════════════════════════════════════════════════════════════════════════════

def load_annotations(path: str, label_col: str = "label") -> dict[str, int]:
    """
    Carga las anotaciones de un CSV.
    Devuelve {id: label_int}.
    Ignora filas con label vacío o no numérico.
    """
    result = {}
    skipped = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frag_id = row.get("id", "").strip()
            label   = row.get(label_col, "").strip()
            if not frag_id:
                continue
            if label not in ("0", "1"):
                skipped += 1
                continue
            result[frag_id] = int(label)
    if skipped:
        print(f"  [AVISO] {skipped} filas sin label válido en {Path(path).name}")
    return result


def load_reference(path: str) -> dict[str, dict]:
    """Carga el archivo de referencia con auto_label y metadata."""
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            frag_id = row.get("id", "").strip()
            if frag_id:
                result[frag_id] = dict(row)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(args):
    print("=" * 65)
    print("  Acuerdo Interanotador — Tarea 2 (binario)")
    print("=" * 65)

    # Cargar
    ann1 = load_annotations(args.ann1)
    ann2 = load_annotations(args.ann2)
    ref  = load_reference(args.ref)

    print(f"  Anotador 1 : {len(ann1)} anotaciones  ({Path(args.ann1).name})")
    print(f"  Anotador 2 : {len(ann2)} anotaciones  ({Path(args.ann2).name})")
    print(f"  Referencia : {len(ref)} fragmentos   ({Path(args.ref).name})")

    # IDs en común
    common_ids = sorted(set(ann1) & set(ann2))
    if not common_ids:
        print("\n[ERROR] No hay IDs en común entre los dos archivos de anotación.")
        print("  Verifica que ambos anotadores usaron el mismo CSV base.")
        sys.exit(1)

    if len(common_ids) < len(ann1) or len(common_ids) < len(ann2):
        print(f"\n  [AVISO] Solo {len(common_ids)} IDs comunes "
              f"(A1={len(ann1)}, A2={len(ann2)}). "
              f"Se calculan métricas sobre los comunes.")

    # Vectores alineados
    la1 = [ann1[i] for i in common_ids]
    la2 = [ann2[i] for i in common_ids]

    # ── Métricas entre anotadores ─────────────────────────────────────────────
    kappa   = cohen_kappa(la1, la2)
    alpha   = krippendorff_alpha(la1, la2)
    pct_agr = percent_agreement(la1, la2)
    matrix  = confusion_matrix_2x2(la1, la2)

    print()
    print("=" * 65)
    print("  MÉTRICAS DE ACUERDO INTERANOTADOR")
    print("=" * 65)
    print(f"  Fragmentos evaluados   : {len(common_ids)}")
    print(f"  Acuerdo simple         : {pct_agr:.1f}%")
    print(f"  Cohen's Kappa (κ)      : {kappa:.4f}  → {kappa_interpretation(kappa)}")
    print(f"  Krippendorff's Alpha   : {alpha:.4f}")
    print()
    req_met = kappa >= 0.60
    print(f"  Requisito κ ≥ 0.60     : {'[OK] CUMPLIDO' if req_met else '[!!] NO CUMPLIDO'}")
    print()
    print("  Matriz de confusión (A1 filas / A2 columnas):")
    print(f"    A1=1 A2=1 (ambos ES_CONTR)        : {matrix['A1=1_A2=1']:>4}")
    print(f"    A1=0 A2=0 (ambos NO_ES_CONTR)     : {matrix['A1=0_A2=0']:>4}")
    print(f"    A1=1 A2=0 (A1=pos, A2=neg)        : {matrix['A1=1_A2=0']:>4}  ← desacuerdo")
    print(f"    A1=0 A2=1 (A1=neg, A2=pos)        : {matrix['A1=0_A2=1']:>4}  ← desacuerdo")

    # ── Acuerdo cada anotador vs auto_label ────────────────────────────────────
    auto_ids = [i for i in common_ids if i in ref and ref[i].get("auto_label", "") in ("0","1")]
    if auto_ids:
        la_auto = [int(ref[i]["auto_label"]) for i in auto_ids]
        la1_vs  = [ann1[i] for i in auto_ids]
        la2_vs  = [ann2[i] for i in auto_ids]

        k1_auto = cohen_kappa(la1_vs, la_auto)
        k2_auto = cohen_kappa(la2_vs, la_auto)

        print()
        print("  Acuerdo vs clasificación automática:")
        print(f"    Anotador 1 vs auto : κ={k1_auto:.4f}  {kappa_interpretation(k1_auto)}")
        print(f"    Anotador 2 vs auto : κ={k2_auto:.4f}  {kappa_interpretation(k2_auto)}")
    else:
        k1_auto = k2_auto = None

    # ── Distribución de etiquetas ─────────────────────────────────────────────
    print()
    print("  Distribución de etiquetas:")
    c1 = Counter(la1); c2 = Counter(la2)
    print(f"    Anotador 1 → 1 (ES):  {c1[1]:>4}  |  0 (NO): {c1[0]:>4}")
    print(f"    Anotador 2 → 1 (ES):  {c2[1]:>4}  |  0 (NO): {c2[0]:>4}")

    # ── Casos de desacuerdo ───────────────────────────────────────────────────
    disagreements = [i for i in common_ids if ann1[i] != ann2[i]]
    print()
    print(f"  Casos de desacuerdo    : {len(disagreements)}/{len(common_ids)} "
          f"({len(disagreements)/len(common_ids)*100:.1f}%)")

    if args.show_disagreements and disagreements:
        print()
        print("  DETALLE DE DESACUERDOS:")
        print("-" * 65)
        for frag_id in disagreements[:30]:   # mostrar máx 30
            meta  = ref.get(frag_id, {})
            snip  = meta.get("snippet", "")[:100].replace("\n", " ")
            auto  = meta.get("auto_label", "?")
            kph   = meta.get("key_phrases", "")[:50]
            print(f"  {frag_id}  A1={ann1[frag_id]}  A2={ann2[frag_id]}  auto={auto}")
            print(f"    frases: {kph}")
            print(f"    texto : {snip}...")
            print()
        if len(disagreements) > 30:
            print(f"  ... y {len(disagreements)-30} más. Ver JSON completo.")

    # ── Etiqueta de consenso (mayoría: si los dos coinciden) ──────────────────
    print()
    print("  ETIQUETA DE CONSENSO para el eval final:")
    consensus = {}
    for frag_id in common_ids:
        if ann1[frag_id] == ann2[frag_id]:
            consensus[frag_id] = ann1[frag_id]
    print(f"    Fragmentos con consenso  : {len(consensus)}/{len(common_ids)}")
    print(f"    Sin consenso (desacuerdo): {len(common_ids)-len(consensus)}")
    if len(common_ids) - len(consensus) > 0:
        print("    → Los fragmentos sin consenso requieren un tercer anotador")
        print("      o resolución por el coordinador del proyecto.")

    # ── Guardar resultado ─────────────────────────────────────────────────────
    result = {
        "n_fragments":       len(common_ids),
        "percent_agreement": pct_agr,
        "cohen_kappa":       kappa,
        "kappa_interpretation": kappa_interpretation(kappa),
        "krippendorff_alpha": alpha,
        "requisito_kappa_060": req_met,
        "confusion_matrix":   matrix,
        "kappa_ann1_vs_auto": k1_auto,
        "kappa_ann2_vs_auto": k2_auto,
        "n_disagreements":   len(disagreements),
        "disagreement_ids":  disagreements,
        "consensus": {
            "n_with_consensus":    len(consensus),
            "n_without_consensus": len(common_ids) - len(consensus),
            "labels": consensus,
        },
        "label_distribution": {
            "annotator1": dict(c1),
            "annotator2": dict(c2),
        },
    }

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 65)
    print(f"  Resultado guardado en: {out_path}")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Calcula acuerdo interanotador (κ, α) para Tarea 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ann1", required=True,
                   help="CSV del Anotador 1 con columna 'label' (0 o 1).")
    p.add_argument("--ann2", required=True,
                   help="CSV del Anotador 2 con columna 'label' (0 o 1).")
    p.add_argument("--ref",  required=True,
                   help="CSV de referencia con columna 'auto_label'.")
    p.add_argument("--out",  default="./acuerdo_interanotador.json",
                   help="Archivo JSON de salida con todas las métricas.")
    p.add_argument("--show-disagreements", action="store_true",
                   help="Imprimir detalle de los fragmentos con desacuerdo.")
    return p.parse_args()


if __name__ == "__main__":
    args = main()
    run(args)
