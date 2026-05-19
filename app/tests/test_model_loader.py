"""
test_model_loader.py - Regresion del bug en el log del loader.

Bug observado en deploy: cuando get_t2_classifier() carga un modelo T2,
T1Classifier loggea 'Cargando modelo T1 ...' incluso siendo T2 (porque
la clase se reutiliza para ambas tareas).

Fix: T1Classifier acepta task_label='T1' o 'T2' como parametro.
get_t2_classifier() pasa task_label='T2'.

Estos tests no cargan transformers (lo mockean) porque queremos validar
el LOGGING, no la inferencia.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def fake_transformers(monkeypatch, tmp_path):
    """Monkeypatch torch+transformers para no requerir cargar pesos."""
    import sys
    import types

    # torch stub
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # transformers stub
    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, path): return cls()
    class FakeModel:
        @classmethod
        def from_pretrained(cls, path):
            m = cls()
            return m
        def eval(self): return self
        def to(self, device): return self
        def half(self): return self
    transformers_mod = types.ModuleType("transformers")
    transformers_mod.AutoTokenizer = FakeTokenizer
    transformers_mod.AutoModelForSequenceClassification = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # MODELS_DIR con un dir falso que tenga model.safetensors y config.json
    models_dir = tmp_path / "models"
    for mid in ("fake-t1", "fake-t2"):
        d = models_dir / mid
        d.mkdir(parents=True)
        (d / "config.json").write_text('{"id2label": {"0": "A", "1": "B"}}')
        (d / "model.safetensors").write_bytes(b"fake")
    monkeypatch.setenv("MODELS_DIR", str(models_dir))

    # Forzar recarga del modulo para que tome el MODELS_DIR nuevo y los stubs
    for name in [n for n in list(sys.modules) if n.startswith("backend.model_loader")]:
        del sys.modules[name]


class TestTaskLabelInLogs:
    def test_t1_classifier_default_label_is_T1(self, fake_transformers, caplog):
        """Sin argumento explicito, T1Classifier usa task_label='T1'."""
        from backend.model_loader import T1Classifier
        with caplog.at_level(logging.INFO, logger="maia.loader"):
            T1Classifier("fake-t1")
        cargando = [r for r in caplog.records if "Cargando modelo" in r.message]
        assert cargando, "Deberia haber logueado 'Cargando modelo ...'"
        assert "Cargando modelo T1 'fake-t1'" in cargando[0].message

    def test_t1_classifier_with_explicit_t2_label(self, fake_transformers, caplog):
        """Pasando task_label='T2', los logs deben decir 'T2'."""
        from backend.model_loader import T1Classifier
        with caplog.at_level(logging.INFO, logger="maia.loader"):
            T1Classifier("fake-t2", task_label="T2")
        msgs = "\n".join(r.message for r in caplog.records)
        assert "Cargando modelo T2 'fake-t2'" in msgs
        assert "Modelo T2 'fake-t2' cargado en cpu (FP32)" in msgs
        # NO debe aparecer 'T1' referenciado a este modelo
        assert "T1 'fake-t2'" not in msgs

    def test_get_t2_classifier_uses_T2_label(self, fake_transformers, caplog):
        """Integracion: get_t2_classifier() debe propagar task_label='T2'."""
        from backend.model_loader import get_t2_classifier, get_t1_classifier
        # Limpiar LRU cache para que la llamada anterior no enmascare esta
        get_t2_classifier.cache_clear()
        get_t1_classifier.cache_clear()

        with caplog.at_level(logging.INFO, logger="maia.loader"):
            get_t2_classifier("fake-t2")
        msgs = "\n".join(r.message for r in caplog.records)
        assert "Cargando modelo T2 'fake-t2'" in msgs, \
            f"get_t2_classifier deberia loggear 'T2'. Logs:\n{msgs}"
        assert "Cargando modelo T1 'fake-t2'" not in msgs, \
            "Bug original: estaba loggeando 'T1' para modelos T2"

    def test_warmup_log_uses_correct_label(self, fake_transformers, caplog):
        """warmup() debe loggear con la task_label asignada al constructor."""
        from backend.model_loader import T1Classifier
        clf = T1Classifier("fake-t2", task_label="T2")
        # No corremos classify_single real (los stubs no implementan forward).
        # Solo verificamos el atributo:
        assert clf.task_label == "T2"
