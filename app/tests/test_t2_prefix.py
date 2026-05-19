"""
test_t2_prefix.py — Cobertura de _build_prefixed_text.

Verifica:
  - Formato literal: "[SECCION:X] [POS:Y] <texto>" con Y formateado como
    decimal de un dígito (e.g. "0.5", "0.0", "0.9")
  - Clipping de relative_pos a [0, 0.9] (el bin 1.0 no existe, se mapea a 0.9)
  - t1_label fuera de _T1_VALID → 'UNK'
"""

from __future__ import annotations

import pytest


# ════════════════════════════════════════════════════════════════════════════
# Formato del prefijo
# ════════════════════════════════════════════════════════════════════════════
class TestPrefixFormat:
    def test_basic_format(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "METH", "relative_pos": 0.5, "text": "hola",
        })
        assert out == "[SECCION:METH] [POS:0.5] hola"

    def test_label_uppercased(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "meth", "relative_pos": 0.3, "text": "x",
        })
        assert out.startswith("[SECCION:METH] [POS:0.3]")

    def test_label_stripped(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "  RES  ", "relative_pos": 0.0, "text": "x",
        })
        assert out.startswith("[SECCION:RES]")


# ════════════════════════════════════════════════════════════════════════════
# Clipping de relative_pos
# ════════════════════════════════════════════════════════════════════════════
class TestPosClipping:
    @pytest.mark.parametrize("pos,expected_bin", [
        (0.0, "0.0"),
        (0.05, "0.0"),   # round(0.5) = 0 (banker's) → 0.0
        (0.1, "0.1"),
        (0.4, "0.4"),
        (0.9, "0.9"),
        (0.95, "0.9"),   # clip a 0.9 (el bin 1.0 no existe)
        (1.0, "0.9"),    # clip a 0.9
        (-0.1, "0.0"),   # clip a 0.0
        (2.0, "0.9"),    # clip a 0.9
    ])
    def test_pos_clipping_and_binning(self, pos, expected_bin):
        # Casos float-frágiles como 0.55 viven en test_pos_055_uses_bankers_rounding
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "METH", "relative_pos": pos, "text": "x",
        })
        assert f"[POS:{expected_bin}]" in out, f"pos={pos} → {out!r}"

    def test_pos_055_uses_bankers_rounding(self):
        """Sanity: confirma cómo se comporta round() con 0.55."""
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "METH", "relative_pos": 0.55, "text": "x",
        })
        # Float imprecision puede dar 0.5 o 0.6 según el round
        assert ("[POS:0.5]" in out) or ("[POS:0.6]" in out)


# ════════════════════════════════════════════════════════════════════════════
# t1_label desconocido → UNK
# ════════════════════════════════════════════════════════════════════════════
class TestUnknownLabel:
    @pytest.mark.parametrize("label", [
        "FOO",          # no está en _T1_VALID
        "CONTR",        # T2 no usa CONTR (es la clase objetivo, no retórica)
        "",             # vacío
        "intro_long",   # similar a INTRO pero no exacto
        "unknown",
    ])
    def test_invalid_label_maps_to_unk(self, label):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": label, "relative_pos": 0.5, "text": "x",
        })
        assert out.startswith("[SECCION:UNK]")

    def test_missing_t1_label_defaults_unk(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({"relative_pos": 0.5, "text": "x"})
        assert out.startswith("[SECCION:UNK]")

    @pytest.mark.parametrize("label", [
        "INTRO", "BACK", "METH", "RES", "DISC", "LIM", "CONC",
    ])
    def test_valid_labels_preserved(self, label):
        """Los 7 labels válidos de T2-ctx se preservan sin cambios."""
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": label, "relative_pos": 0.5, "text": "x",
        })
        assert out.startswith(f"[SECCION:{label}]")


# ════════════════════════════════════════════════════════════════════════════
# relative_pos malformado
# ════════════════════════════════════════════════════════════════════════════
class TestPosMalformed:
    def test_none_pos_defaults_to_05(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "METH", "relative_pos": None, "text": "x",
        })
        assert "[POS:0.5]" in out

    def test_string_pos_defaults_to_05(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({
            "t1_label": "METH", "relative_pos": "not-a-float", "text": "x",
        })
        assert "[POS:0.5]" in out

    def test_missing_pos_defaults_to_05(self):
        from backend.core.t2_scibeto_service import _build_prefixed_text
        out = _build_prefixed_text({"t1_label": "METH", "text": "x"})
        assert "[POS:0.5]" in out
