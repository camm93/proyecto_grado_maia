"""
test_contributions.py — Cobertura del endpoint /api/contributions.

Verifica:
  - Cada model_id despacha al servicio correcto (con monkeypatch de los
    servicios para no requerir torch/openai)
  - Fallback transparente a heurística ante ModelNotAvailableError,
    con el nombre marcado [fallback heurístico: ...]
"""

from __future__ import annotations

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────
def _make_fragment(idx=0, text="Proponemos un nuevo método de XYZ."):
    """Fragmento mínimamente válido para el schema."""
    return {
        "index": idx,
        "text": text,
        "t1_label": "METH",
        "t1_label_name": "Metodología",
        "t1_color": "#abc",
        "t1_confidence": 0.8,
        "rhetorical_zone": "core",
        "char_start": 0,
        "char_end": len(text),
        "word_count": len(text.split()),
        "relative_pos": 0.5,
        "low_confidence": False,
    }


def _fake_classifier_result(idx, is_contrib, conf=0.8):
    """Schema que devuelven los servicios run_t2_*."""
    return {
        "index": idx,
        "is_contribution": is_contrib,
        "confidence": conf,
        "label": "ES_CONTRIBUCION" if is_contrib else "NO_ES_CONTRIBUCION",
        "t1_label": "METH",
        "t1_label_name": "Metodología",
        "t1_color": "#abc",
        "t1_confidence": 0.8,
        "rhetorical_zone": "core",
        "rhetorical_context": "",
        "char_start": 0,
        "char_end": 30,
        "word_count": 5,
        "relative_pos": 0.5,
    }


# ════════════════════════════════════════════════════════════════════════════
# Dispatch SciBETO (con info["available"]=True monkeypatcheado)
# ════════════════════════════════════════════════════════════════════════════
class TestScibetoDispatch:
    @pytest.fixture
    def client(self, monkeypatch):
        """TestClient con scibeto-es-t2 marcado available + run_t2_scibeto mock."""
        from backend.core import catalog as cat_mod

        # Forzar `available=True` para los modelos del test:
        # 1. Para encoders SciBETO: mockear _local_weights_present
        # 2. Para LLMs Llama: mockear _llama_available
        # 3. Para LLMs GPT: setear OPENAI_API_KEY en env
        # Esto sobrevive a refresh_catalog() que ahora se llama en cada request.
        monkeypatch.setattr(cat_mod, "_local_weights_present", lambda mid: True)
        monkeypatch.setattr(cat_mod, "_llama_available", lambda: True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        # available=True como fallback inicial (refresh_catalog mantendrá esto)
        for m in cat_mod.MODELS_T2:
            if m["id"] in ("scibeto-es-t2", "scibeto-es-t2-ctx"):
                m["available"] = True

        called = {"scibeto": False, "llm": False, "heur": False}

        def fake_scibeto(fragments, model_id="scibeto-es-t2"):
            called["scibeto"] = True
            return [_fake_classifier_result(f["index"], True) for f in fragments]

        def fake_llm(model_id, fragments):
            called["llm"] = True
            return [_fake_classifier_result(f["index"], False) for f in fragments]

        def fake_heur(fragments):
            called["heur"] = True
            return [_fake_classifier_result(f["index"], False) for f in fragments]

        # Monkeypatch en el módulo donde se IMPORTA, no donde se define
        import backend.api.contributions as ct
        monkeypatch.setattr(ct, "run_t2_scibeto", fake_scibeto)
        monkeypatch.setattr(ct, "run_t2_llm", fake_llm)
        monkeypatch.setattr(ct, "run_t2_heuristic", fake_heur)

        from fastapi.testclient import TestClient
        from backend.main import app
        c = TestClient(app)
        yield c, called
        # Cleanup: restaurar available=False para no contaminar otros tests
        for m in cat_mod.MODELS_T2:
            if m["id"] in ("scibeto-es-t2", "scibeto-es-t2-ctx"):
                m["available"] = False

    def test_scibeto_es_t2_dispatches_to_scibeto(self, client):
        c, called = client
        r = c.post("/api/contributions", json={
            "fragments": [_make_fragment()], "model_id": "scibeto-es-t2",
        })
        assert r.status_code == 200
        assert called["scibeto"] is True
        assert called["llm"] is False
        assert called["heur"] is False
        assert r.json()["mode"] == "encoder"

    def test_scibeto_es_t2_ctx_dispatches_to_scibeto(self, client):
        c, called = client
        r = c.post("/api/contributions", json={
            "fragments": [_make_fragment()], "model_id": "scibeto-es-t2-ctx",
        })
        assert r.status_code == 200
        assert called["scibeto"] is True

    def test_heuristic_id_dispatches_to_heuristic(self, client):
        c, called = client
        r = c.post("/api/contributions", json={
            "fragments": [_make_fragment()], "model_id": "heuristic",
        })
        assert r.status_code == 200
        assert called["heur"] is True
        assert called["scibeto"] is False


# ════════════════════════════════════════════════════════════════════════════
# Dispatch LLMs (los 4 IDs)
# ════════════════════════════════════════════════════════════════════════════
class TestLlmDispatch:
    @pytest.fixture
    def client(self, monkeypatch):
        from backend.core import catalog as cat_mod
        # Para que LLMs queden available=True después de refresh_catalog:
        # mockear _llama_available y setear OPENAI_API_KEY
        monkeypatch.setattr(cat_mod, "_llama_available", lambda: True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        # Marcar los 4 LLMs como disponibles (estado inicial)
        for m in cat_mod.MODELS_T2:
            if m["id"] in {
                "llama-3.1-8b-t2", "llama-3.1-8b-t2-fs8",
                "gpt-4o-mini-t2", "gpt-4o-mini-t2-fs8",
            }:
                m["available"] = True

        called = {"llm_calls": []}

        def fake_llm(model_id, fragments):
            called["llm_calls"].append(model_id)
            return [_fake_classifier_result(f["index"], False) for f in fragments]

        import backend.api.contributions as ct
        monkeypatch.setattr(ct, "run_t2_llm", fake_llm)

        from fastapi.testclient import TestClient
        from backend.main import app
        c = TestClient(app)
        yield c, called
        for m in cat_mod.MODELS_T2:
            if m["id"] in {
                "llama-3.1-8b-t2", "llama-3.1-8b-t2-fs8",
                "gpt-4o-mini-t2", "gpt-4o-mini-t2-fs8",
            }:
                m["available"] = False

    @pytest.mark.parametrize("model_id", [
        "llama-3.1-8b-t2",
        "llama-3.1-8b-t2-fs8",
        "gpt-4o-mini-t2",
        "gpt-4o-mini-t2-fs8",
    ])
    def test_each_llm_id_dispatches_correctly(self, client, model_id):
        c, called = client
        r = c.post("/api/contributions", json={
            "fragments": [_make_fragment()], "model_id": model_id,
        })
        assert r.status_code == 200
        assert called["llm_calls"] == [model_id]


# ════════════════════════════════════════════════════════════════════════════
# Fallback transparente
# ════════════════════════════════════════════════════════════════════════════
class TestFallbackToHeuristic:
    def test_scibeto_fallback_when_model_not_available_error(self, monkeypatch):
        """Si run_t2_scibeto levanta ModelNotAvailableError → heurística
        con nombre marcado [fallback heurístico: ...]."""
        from backend.core import catalog as cat_mod
        from backend.model_loader import ModelNotAvailableError

        # Mockear _local_weights_present para que refresh_catalog mantenga
        # available=True para scibeto-es-t2 (queremos llegar al run_t2_scibeto
        # para verificar que su ModelNotAvailableError dispare el fallback)
        monkeypatch.setattr(cat_mod, "_local_weights_present", lambda mid: True)
        for m in cat_mod.MODELS_T2:
            if m["id"] == "scibeto-es-t2":
                m["available"] = True

        def raise_not_available(*a, **kw):
            raise ModelNotAvailableError("pesos no encontrados")

        heur_called = {"ok": False}

        def fake_heur(fragments):
            heur_called["ok"] = True
            return [_fake_classifier_result(f["index"], False) for f in fragments]

        import backend.api.contributions as ct
        monkeypatch.setattr(ct, "run_t2_scibeto", raise_not_available)
        monkeypatch.setattr(ct, "run_t2_heuristic", fake_heur)

        from fastapi.testclient import TestClient
        from backend.main import app
        c = TestClient(app)
        r = c.post("/api/contributions", json={
            "fragments": [_make_fragment()], "model_id": "scibeto-es-t2",
        })
        assert r.status_code == 200
        assert heur_called["ok"] is True
        body = r.json()
        assert body["mode"] == "heuristic"
        assert "fallback heuristico" in body["model_name"]

        # Cleanup
        for m in cat_mod.MODELS_T2:
            if m["id"] == "scibeto-es-t2":
                m["available"] = False

    def test_unavailable_model_falls_back_silently(self, monkeypatch):
        """Modelo no disponible (info[available]=False) sin tirar excepción
        → heurística sin reportarlo como error."""
        heur_called = {"ok": False}

        def fake_heur(fragments):
            heur_called["ok"] = True
            return [_fake_classifier_result(f["index"], False) for f in fragments]

        import backend.api.contributions as ct
        monkeypatch.setattr(ct, "run_t2_heuristic", fake_heur)

        from fastapi.testclient import TestClient
        from backend.main import app
        c = TestClient(app)
        # scibeto-es-t2 NO está available por default en el catálogo
        r = c.post("/api/contributions", json={
            "fragments": [_make_fragment()], "model_id": "scibeto-es-t2",
        })
        assert r.status_code == 200
        assert heur_called["ok"] is True
