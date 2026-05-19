"""
test_catalog.py — Cobertura del catálogo de modelos.

Verifica:
  - Los 7 model_ids T2 con sus familias correctas (incluye el rename T1
    a scibeto-es-t1 del issue #1)
  - _local_weights_present() excluye training_args.bin (metadata del
    Trainer, no peso)
  - _llama_available() respeta LLAMA_ENABLED=false
"""

from __future__ import annotations

import pytest


# ════════════════════════════════════════════════════════════════════════════
# Catálogo: model_ids esperados
# ════════════════════════════════════════════════════════════════════════════
class TestT2Catalog:
    def test_t2_has_seven_models(self):
        from backend.core.catalog import MODELS_T2
        assert len(MODELS_T2) == 7

    def test_t2_model_ids_and_families(self):
        """Los 7 IDs y sus familias deben ser exactamente los esperados."""
        from backend.core.catalog import MODELS_T2
        expected = {
            "heuristic":              "heuristic",
            "scibeto-es-t2":          "encoder",
            "scibeto-es-t2-ctx":      "encoder",
            "llama-3.1-8b-t2":        "llm_open",
            "llama-3.1-8b-t2-fs8":    "llm_open",
            "gpt-4o-mini-t2":         "llm_api",
            "gpt-4o-mini-t2-fs8":     "llm_api",
        }
        got = {m["id"]: m["family"] for m in MODELS_T2}
        assert got == expected

    def test_all_t2_models_have_task_T2(self):
        from backend.core.catalog import MODELS_T2
        assert all(m["task"] == "T2" for m in MODELS_T2)

    def test_heuristic_t2_is_always_available(self):
        from backend.core.catalog import MODELS_T2
        h = next(m for m in MODELS_T2 if m["id"] == "heuristic")
        assert h["available"] is True


class TestT1Catalog:
    def test_t1_id_is_scibeto_es_t1(self):
        """v6: el ID del encoder T1 cambio de 'scibeto-v5-cris' a
        'scibeto-es-t1' (nuevo modelo re-entrenado, run task1-scibeto-v12-cris).
        Anteriormente (Issue #1) habia sido renombrado de 'scibert-es' a
        'scibeto-v5-cris'; ahora se actualiza al nombre final 'scibeto-es-t1'
        que es consistente con scibeto-es-t2 (T2 baseline)."""
        from backend.core.catalog import MODELS_T1
        ids = {m["id"] for m in MODELS_T1}
        assert "scibeto-es-t1" in ids
        assert "scibert-es" not in ids
        assert "scibeto-v5-cris" not in ids  # deprecated, ya no en catalogo

    def test_t1_zs_variants_present(self):
        """v7: notebook task1_edwin_v3 evaluo ZS y FS para Llama y GPT.
        El catalogo expone las 4 variantes LLM en T1 (2 FS + 2 ZS)."""
        from backend.core.catalog import MODELS_T1
        ids = {m["id"] for m in MODELS_T1}
        assert "llama-3.1-8b-t1" in ids
        assert "llama-3.1-8b-t1-zs" in ids
        assert "gpt-4o-mini-t1" in ids
        assert "gpt-4o-mini-t1-zs" in ids

    def test_t1_has_six_ids_total(self):
        """v7: heuristic + scibeto-es-t1 + 4 LLMs = 6 modelos."""
        from backend.core.catalog import MODELS_T1
        ids = {m["id"] for m in MODELS_T1}
        assert ids == {
            "heuristic", "scibeto-es-t1",
            "llama-3.1-8b-t1", "llama-3.1-8b-t1-zs",
            "gpt-4o-mini-t1", "gpt-4o-mini-t1-zs",
        }

    def test_t1_zs_variants_share_family_with_fs(self):
        """ZS y FS de la misma familia deben tener el mismo `family` (llm_open
        o llm_api). Asi el frontend los agrupa en el mismo optgroup."""
        from backend.core.catalog import MODELS_T1
        cat = {m["id"]: m for m in MODELS_T1}
        assert cat["llama-3.1-8b-t1"]["family"] == cat["llama-3.1-8b-t1-zs"]["family"] == "llm_open"
        assert cat["gpt-4o-mini-t1"]["family"] == cat["gpt-4o-mini-t1-zs"]["family"] == "llm_api"


# ════════════════════════════════════════════════════════════════════════════
# _local_weights_present()
# ════════════════════════════════════════════════════════════════════════════
class TestLocalWeightsPresent:
    @pytest.fixture
    def models_root(self, tmp_path, monkeypatch):
        d = tmp_path / "models"
        d.mkdir()
        monkeypatch.setenv("MODELS_DIR", str(d))
        return d

    def test_returns_true_with_safetensors(self, models_root):
        from backend.core.catalog import _local_weights_present
        mp = models_root / "myid"
        mp.mkdir()
        (mp / "config.json").write_text("{}")
        (mp / "model.safetensors").write_bytes(b"x")
        assert _local_weights_present("myid") is True

    def test_returns_true_with_pytorch_bin(self, models_root):
        from backend.core.catalog import _local_weights_present
        mp = models_root / "myid"
        mp.mkdir()
        (mp / "config.json").write_text("{}")
        (mp / "pytorch_model.bin").write_bytes(b"x")
        assert _local_weights_present("myid") is True

    def test_returns_false_without_config(self, models_root):
        from backend.core.catalog import _local_weights_present
        mp = models_root / "myid"
        mp.mkdir()
        (mp / "model.safetensors").write_bytes(b"x")
        assert _local_weights_present("myid") is False

    def test_training_args_bin_does_not_count_as_weights(self, models_root):
        """training_args.bin es metadata del Trainer, NO peso. Si está
        solo, _local_weights_present debe devolver False."""
        from backend.core.catalog import _local_weights_present
        mp = models_root / "myid"
        mp.mkdir()
        (mp / "config.json").write_text("{}")
        (mp / "training_args.bin").write_bytes(b"trainer-metadata")
        # No hay model.safetensors ni pytorch_model.bin
        assert _local_weights_present("myid") is False

    def test_returns_false_when_dir_missing(self, models_root):
        from backend.core.catalog import _local_weights_present
        assert _local_weights_present("never-existed") is False

    def test_supports_sharded_safetensors(self, models_root):
        from backend.core.catalog import _local_weights_present
        mp = models_root / "shard-model"
        mp.mkdir()
        (mp / "config.json").write_text("{}")
        (mp / "model-00001-of-00003.safetensors").write_bytes(b"x")
        assert _local_weights_present("shard-model") is True


# ════════════════════════════════════════════════════════════════════════════
# _llama_available()
# ════════════════════════════════════════════════════════════════════════════
class TestLlamaAvailable:
    def test_returns_false_without_hf_token(self, monkeypatch):
        from backend.core.catalog import _llama_available
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert _llama_available() is False

    def test_returns_true_with_token_and_default_enabled(self, monkeypatch):
        from backend.core.catalog import _llama_available
        monkeypatch.setenv("HF_TOKEN", "hf_fake")
        monkeypatch.delenv("LLAMA_ENABLED", raising=False)
        assert _llama_available() is True

    def test_respects_llama_enabled_false(self, monkeypatch):
        """LLAMA_ENABLED=false fuerza no-disponible incluso con HF_TOKEN."""
        from backend.core.catalog import _llama_available
        monkeypatch.setenv("HF_TOKEN", "hf_fake")
        monkeypatch.setenv("LLAMA_ENABLED", "false")
        assert _llama_available() is False

    def test_llama_enabled_false_is_case_insensitive(self, monkeypatch):
        from backend.core.catalog import _llama_available
        monkeypatch.setenv("HF_TOKEN", "hf_fake")
        monkeypatch.setenv("LLAMA_ENABLED", "FALSE")
        assert _llama_available() is False
