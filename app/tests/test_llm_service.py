"""
test_llm_service.py — Cobertura del módulo LLM (T2 con Llama y GPT).

Verifica:
  - SHA256 de FEW_SHOT_EXAMPLES coincide con el manifest
    (4ccc1238b0afcc0b3885d818d911881ce8f372d9c17ed6bc43b918d14e20d509)
  - parse_llm_answer() para edge cases: "Sí", "Si.", "NO!", "Maybe",
    "", None
  - LLM_DISPATCH tiene los 4 IDs correctos con (backend, few_shot)
"""

from __future__ import annotations

import hashlib
import json

import pytest


# ════════════════════════════════════════════════════════════════════════════
# SHA256 del few-shot
# ════════════════════════════════════════════════════════════════════════════
class TestFewShotSha256:
    EXPECTED_SHA = (
        "4ccc1238b0afcc0b3885d818d911881ce8f372d9c17ed6bc43b918d14e20d509"
    )

    def test_few_shot_count_is_eight(self):
        from backend.core.llm_service import FEW_SHOT_EXAMPLES
        assert len(FEW_SHOT_EXAMPLES) == 8

    def test_few_shot_balance_four_si_four_no(self):
        from backend.core.llm_service import FEW_SHOT_EXAMPLES
        labels = [lbl for lbl, _ in FEW_SHOT_EXAMPLES]
        assert labels.count("SI") == 4
        assert labels.count("NO") == 4

    def test_few_shot_alternated_order(self):
        """Orden esperado: SI, NO, SI, NO, SI, NO, SI, NO."""
        from backend.core.llm_service import FEW_SHOT_EXAMPLES
        labels = [lbl for lbl, _ in FEW_SHOT_EXAMPLES]
        assert labels == ["SI", "NO"] * 4

    def test_few_shot_sha256_matches_manifest(self):
        """SHA bit-a-bit del set few-shot publicado en
        release_manifest.json."""
        from backend.core.llm_service import FEW_SHOT_EXAMPLES
        canonical = json.dumps(
            [{"label": lbl, "text": txt} for lbl, txt in FEW_SHOT_EXAMPLES],
            ensure_ascii=False, sort_keys=True,
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == self.EXPECTED_SHA


# ════════════════════════════════════════════════════════════════════════════
# parse_llm_answer — edge cases
# ════════════════════════════════════════════════════════════════════════════
class TestParseLlmAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("SI", 1),
        ("Sí", 1),
        ("Si.", 1),
        ("si, claramente", 1),
        ("sí — es contribución", 1),
        ("YES", 1),
        ("yes, this is", 1),
        ("NO", 0),
        ("NO!", 0),
        ("no, no es", 0),
        ("No.", 0),
    ])
    def test_clear_answers(self, text, expected):
        from backend.core.llm_service import parse_llm_answer
        assert parse_llm_answer(text) == expected

    @pytest.mark.parametrize("text", [
        "Maybe",
        "tal vez",
        "no estoy seguro",  # empieza con "no" → 0, no es ambiguo realmente
        "",
        "...",
        "Unknown response",
    ])
    def test_ambiguous_defaults_to_zero(self, text):
        """Política: cualquier respuesta no clara → 0 (NO).
        Esto se alinea con el comportamiento conservador (recall pos bajo
        pero precision alta cuando responde SI)."""
        from backend.core.llm_service import parse_llm_answer
        # Casos que comienzan con "no" devuelven 0 explícitamente; los demás
        # caen al default 0.
        assert parse_llm_answer(text) == 0

    def test_none_input(self):
        from backend.core.llm_service import parse_llm_answer
        assert parse_llm_answer(None) == 0

    def test_non_string_input(self):
        from backend.core.llm_service import parse_llm_answer
        assert parse_llm_answer(42) == 0
        assert parse_llm_answer(["si"]) == 0

    def test_strips_whitespace(self):
        from backend.core.llm_service import parse_llm_answer
        assert parse_llm_answer("  SI  ") == 1
        assert parse_llm_answer("\n\tno\n") == 0


# ════════════════════════════════════════════════════════════════════════════
# LLM_DISPATCH
# ════════════════════════════════════════════════════════════════════════════
class TestLlmDispatch:
    def test_dispatch_has_four_entries(self):
        from backend.core.llm_service import LLM_DISPATCH
        assert len(LLM_DISPATCH) == 4

    def test_dispatch_exact_keys(self):
        from backend.core.llm_service import LLM_DISPATCH
        expected_keys = {
            "llama-3.1-8b-t2",
            "llama-3.1-8b-t2-fs8",
            "gpt-4o-mini-t2",
            "gpt-4o-mini-t2-fs8",
        }
        assert set(LLM_DISPATCH.keys()) == expected_keys

    def test_zero_shot_entries_have_few_shot_false(self):
        from backend.core.llm_service import LLM_DISPATCH
        assert LLM_DISPATCH["llama-3.1-8b-t2"] == ("llama", False)
        assert LLM_DISPATCH["gpt-4o-mini-t2"] == ("gpt", False)

    def test_few_shot_entries_have_few_shot_true(self):
        from backend.core.llm_service import LLM_DISPATCH
        assert LLM_DISPATCH["llama-3.1-8b-t2-fs8"] == ("llama", True)
        assert LLM_DISPATCH["gpt-4o-mini-t2-fs8"] == ("gpt", True)


# ════════════════════════════════════════════════════════════════════════════
# TAREA 1 — Llama T1 + GPT T1 (few-shot k=14)
# ════════════════════════════════════════════════════════════════════════════
class TestT1FewShotSha256:
    """SHA del few-shot T1 (14 ejemplos, 2 por clase x 7 clases).
    Debe coincidir bit-a-bit con el SHA publicado en el manifest."""
    EXPECTED_SHA = (
        "7d154afd9c19699d1c637f2b98028eff964fbb8e4d3bc4f06005d6093387cec0"
    )

    def test_few_shot_count_is_14(self):
        from backend.core.llm_service import FEW_SHOT_EXAMPLES_T1
        assert len(FEW_SHOT_EXAMPLES_T1) == 14

    def test_two_per_class(self):
        from backend.core.llm_service import FEW_SHOT_EXAMPLES_T1, T1_LABELS
        from collections import Counter
        labels = [lbl for lbl, _ in FEW_SHOT_EXAMPLES_T1]
        c = Counter(labels)
        assert set(c.keys()) == set(T1_LABELS), \
            f"Etiquetas esperadas {T1_LABELS}, encontradas {list(c.keys())}"
        for lbl, cnt in c.items():
            assert cnt == 2, f"Clase {lbl} tiene {cnt} ejemplos, debe ser 2"

    def test_sha256_matches_manifest(self):
        from backend.core.llm_service import FEW_SHOT_EXAMPLES_T1
        canonical = json.dumps(
            [{"label": lbl, "text": txt} for lbl, txt in FEW_SHOT_EXAMPLES_T1],
            ensure_ascii=False, sort_keys=True,
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == self.EXPECTED_SHA


class TestParseT1Label:
    """Parser tolerante T1: acepta sigla, variantes en español, JSON."""
    def test_canonical_siglas(self):
        from backend.core.llm_service import parse_llm_label_t1
        for sigla in ["INTRO", "BACK", "METH", "RES", "DISC", "LIM", "CONC"]:
            assert parse_llm_label_t1(sigla) == sigla

    def test_spanish_variants(self):
        from backend.core.llm_service import parse_llm_label_t1
        assert parse_llm_label_t1("Introducción") == "INTRO"
        assert parse_llm_label_t1("Metodología") == "METH"
        assert parse_llm_label_t1("Discusión") == "DISC"
        assert parse_llm_label_t1("Conclusiones") == "CONC"
        assert parse_llm_label_t1("Antecedentes") == "BACK"
        assert parse_llm_label_t1("Limitaciones") == "LIM"

    def test_json_mode(self):
        """GPT con JSON mode devuelve {"category": "..."}."""
        from backend.core.llm_service import parse_llm_label_t1
        assert parse_llm_label_t1('{"category": "INTRO"}') == "INTRO"
        assert parse_llm_label_t1('{"category": "Metodología"}') == "METH"

    def test_priority_longer_first(self):
        """'INTRODUCCIÓN' debe matchear antes que 'INTRO'."""
        from backend.core.llm_service import parse_llm_label_t1
        # En la práctica los dos devuelven INTRO; el test es que ningún
        # match parcial corrompa el resultado.
        assert parse_llm_label_t1("INTRODUCCIÓN") == "INTRO"

    def test_empty_or_garbage_returns_unknown(self):
        from backend.core.llm_service import parse_llm_label_t1
        assert parse_llm_label_t1("") == "UNKNOWN"
        assert parse_llm_label_t1(None) == "UNKNOWN"
        assert parse_llm_label_t1("no sé") == "UNKNOWN"
        assert parse_llm_label_t1("xyz123") == "UNKNOWN"

    def test_contr_returns_unknown_for_t1(self):
        """CONTR no es una clase T1 válida (T1 tiene 7 sin CONTR)."""
        from backend.core.llm_service import parse_llm_label_t1
        # _LLM_LABEL_MAP_T1 mapea CONTR pero el filtro final lo descarta
        assert parse_llm_label_t1("CONTR") == "UNKNOWN"
        assert parse_llm_label_t1("Contribución") == "UNKNOWN"


class TestLlmDispatchT1:
    def test_dispatch_keys_present(self):
        """v7: 4 entradas T1 LLM (2 FS-14 + 2 ZS) tras task1_edwin_v3."""
        from backend.core.llm_service import LLM_DISPATCH_T1
        assert set(LLM_DISPATCH_T1.keys()) == {
            "llama-3.1-8b-t1", "llama-3.1-8b-t1-zs",
            "gpt-4o-mini-t1", "gpt-4o-mini-t1-zs",
        }

    def test_dispatch_fs_variants_have_few_shot_true(self):
        """Los IDs sin sufijo (compat con v6) mantienen few-shot=True."""
        from backend.core.llm_service import LLM_DISPATCH_T1
        assert LLM_DISPATCH_T1["llama-3.1-8b-t1"][1] is True
        assert LLM_DISPATCH_T1["gpt-4o-mini-t1"][1] is True

    def test_dispatch_zs_variants_have_few_shot_false(self):
        """Los IDs con sufijo -zs son zero-shot (sin bloque FEW_SHOT_BLOCK_T1)."""
        from backend.core.llm_service import LLM_DISPATCH_T1
        assert LLM_DISPATCH_T1["llama-3.1-8b-t1-zs"][1] is False
        assert LLM_DISPATCH_T1["gpt-4o-mini-t1-zs"][1] is False

    def test_dispatch_backends_correct(self):
        """El primer elemento del tuple es el backend correcto."""
        from backend.core.llm_service import LLM_DISPATCH_T1
        assert LLM_DISPATCH_T1["llama-3.1-8b-t1"][0] == "llama"
        assert LLM_DISPATCH_T1["llama-3.1-8b-t1-zs"][0] == "llama"
        assert LLM_DISPATCH_T1["gpt-4o-mini-t1"][0] == "gpt"
        assert LLM_DISPATCH_T1["gpt-4o-mini-t1-zs"][0] == "gpt"

    def test_run_t1_llm_unknown_id_raises(self):
        from backend.core.llm_service import run_t1_llm
        from backend.model_loader import ModelNotAvailableError
        with pytest.raises(ModelNotAvailableError, match="desconocido"):
            run_t1_llm("modelo-inexistente", ["snippet"])
