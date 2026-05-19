"""
test_few_shots_endpoint.py — Tests del endpoint GET /api/few_shots.

Verifica que el endpoint devuelva los datasets T1 (k=14) y T2 (k=8) con
estructura correcta y que los SHA256 declarados coincidan con los datos
servidos (paridad bit-a-bit con el manifest publicado).

Tests:
  - GET /api/few_shots devuelve 200 OK con shape {T1: {...}, T2: {...}}
  - T1.k == 14 y T2.k == 8
  - T1.examples tiene 2 ejemplos por clase x 7 clases
  - T2.examples tiene 4 SI + 4 NO con orden alternado
  - SHA256 declarado coincide con el calculado sobre el campo `examples`
"""

from __future__ import annotations

import hashlib
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Cliente FastAPI montando el router de system.py."""
    from fastapi import FastAPI
    from backend.api import system as system_api
    app = FastAPI()
    app.include_router(system_api.router)
    return TestClient(app)


class TestFewShotsEndpoint:

    def test_returns_200(self, client):
        r = client.get("/api/few_shots")
        assert r.status_code == 200

    def test_has_t1_and_t2_keys(self, client):
        d = client.get("/api/few_shots").json()
        assert "T1" in d and "T2" in d

    def test_t1_has_14_examples(self, client):
        d = client.get("/api/few_shots").json()
        assert d["T1"]["k"] == 14
        assert len(d["T1"]["examples"]) == 14

    def test_t2_has_8_examples(self, client):
        d = client.get("/api/few_shots").json()
        assert d["T2"]["k"] == 8
        assert len(d["T2"]["examples"]) == 8

    def test_t1_two_per_class(self, client):
        from collections import Counter
        d = client.get("/api/few_shots").json()
        labels = [ex["label"] for ex in d["T1"]["examples"]]
        c = Counter(labels)
        assert set(c.keys()) == {"INTRO", "BACK", "METH", "RES", "DISC", "LIM", "CONC"}
        for lbl, cnt in c.items():
            assert cnt == 2, f"T1: clase {lbl} tiene {cnt} ejemplos, debe ser 2"

    def test_t2_balance_four_si_four_no(self, client):
        from collections import Counter
        d = client.get("/api/few_shots").json()
        labels = [ex["label"] for ex in d["T2"]["examples"]]
        c = Counter(labels)
        assert c["SI"] == 4
        assert c["NO"] == 4

    def test_t2_alternated_order(self, client):
        """El set k=8 de T2 debe ir SI, NO, SI, NO, ... para neutralizar
        sesgo posicional del LLM."""
        d = client.get("/api/few_shots").json()
        labels = [ex["label"] for ex in d["T2"]["examples"]]
        assert labels == ["SI", "NO", "SI", "NO", "SI", "NO", "SI", "NO"]

    def test_t1_sha256_matches_data(self, client):
        """El SHA256 declarado debe coincidir con el hash canonical de los
        examples servidos (mismo algoritmo que test_llm_service.py)."""
        d = client.get("/api/few_shots").json()
        canonical = json.dumps(
            [{"label": ex["label"], "text": ex["text"]}
             for ex in d["T1"]["examples"]],
            ensure_ascii=False, sort_keys=True,
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == d["T1"]["sha256"]

    def test_t2_sha256_matches_data(self, client):
        d = client.get("/api/few_shots").json()
        canonical = json.dumps(
            [{"label": ex["label"], "text": ex["text"]}
             for ex in d["T2"]["examples"]],
            ensure_ascii=False, sort_keys=True,
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == d["T2"]["sha256"]

    def test_examples_have_label_and_text(self, client):
        d = client.get("/api/few_shots").json()
        for task in ("T1", "T2"):
            for ex in d[task]["examples"]:
                assert "label" in ex and isinstance(ex["label"], str)
                assert "text" in ex and isinstance(ex["text"], str)
                assert len(ex["text"]) > 0
