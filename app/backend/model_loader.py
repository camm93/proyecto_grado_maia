"""
model_loader.py — Carga de modelos fine-tuned
===================================================
Gestiona la carga perezosa (lazy) de los encoders entrenados.
Los pesos (~650 MB - 1.3 GB por modelo) se leen desde MODELS_DIR
una sola vez y se cachean en memoria durante la vida del proceso.

Uso T2 (binaria):
    from model_loader import get_predictor
    predictor = get_predictor("scibeto-es-t2")
    result = predictor(text)  # → [{"label": "ES_CONTRIBUCION", "score": 0.92}]

Uso T1 (multiclase, devuelve todas las probabilidades):
    from model_loader import get_t1_classifier
    clf = get_t1_classifier("scibeto-es-t1")
    result = clf(text)  # → {"label": "METH", "confidence": 0.61,
                        #     "alternatives": [{"label": "RES", "score": 0.21}, ...],
                        #     "all_probs": {"INTRO": 0.05, "BACK": 0.08, ...}}
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable

log = logging.getLogger("maia.loader")

MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


class ModelNotAvailableError(Exception):
    pass


# ── Helpers comunes ─────────────────────────────────────────────────────────
def _verify_weights(model_path: Path) -> None:
    """Verifica que el directorio tenga config.json y pesos REALES del modelo.

    `training_args.bin` y `optimizer.bin` son artifacts del Trainer y NO
    cuentan como pesos del encoder. Se buscan solo los nombres canónicos
    de HF (pytorch_model.bin, model.safetensors, o variantes sharded).
    """
    if not model_path.exists():
        raise ModelNotAvailableError(
            f"Modelo no encontrado en {model_path}. "
            f"Descarga los pesos y colocalos alli."
        )
    if not (model_path / "config.json").exists():
        raise ModelNotAvailableError(
            f"Falta config.json en {model_path}."
        )
    weights = (
        list(model_path.glob("pytorch_model.bin"))
        + list(model_path.glob("pytorch_model-*-of-*.bin"))
        + list(model_path.glob("model.safetensors"))
        + list(model_path.glob("model-*-of-*.safetensors"))
    )
    if not weights:
        raise ModelNotAvailableError(
            f"No se encontraron pesos del modelo en {model_path}. "
            "Esperado: pytorch_model.bin o model.safetensors "
            "(training_args.bin no son pesos del encoder)."
        )


def _read_id2label(model_path: Path) -> dict[int, str]:
    """Lee id2label desde config.json (siempre presente en modelos HF)."""
    with open(model_path / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    raw = config.get("id2label", {})
    return {int(k): v for k, v in raw.items()}


# ── T2: pipeline de clasificación binaria (legacy, no tocar) ────────────────
@lru_cache(maxsize=4)
def get_predictor(model_id: str):
    """
    Carga y cachea un predictor de clasificación binaria (T2).
    Devuelve un pipeline de HuggingFace que acepta texto y produce
    {'label': str, 'score': float}.
    """
    model_path = MODELS_DIR / model_id
    _verify_weights(model_path)

    try:
        from transformers import pipeline
        log.info(f"Cargando modelo T2 '{model_id}' desde {model_path} ...")
        predictor = pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            device=-1,
            truncation=True,
            max_length=512,
        )
        log.info(f"Modelo T2 '{model_id}' cargado OK")
        return predictor
    except ImportError:
        raise ModelNotAvailableError(
            "transformers no esta instalado. "
            "Descomentar en requirements.txt: transformers>=4.45.0 y torch>=2.4.0"
        )
    except Exception as e:
        raise ModelNotAvailableError(f"Error al cargar T2 '{model_id}': {e}")


# ── T1: clasificador multiclase con probabilidades completas + batching ────
class T1Classifier:
    """
    Wrapper del modelo T1 fine-tuned con soporte para batch.

    Métodos públicos:
        classify_single(text)           → dict con label, confidence, alternatives, all_probs
        classify_batch(texts, micro_bs) → lista de dicts, uno por texto

    El batching es crítico para latencia: clasificar 80 párrafos uno por uno
    en CPU puede tardar 5-10 minutos; en batch de 32 el mismo trabajo toma
    1-2 min. En GPU el speedup es aún mayor (~50x sobre el loop secuencial).

    device se determina por (en orden):
        1. argumento explícito `device`
        2. variable de entorno T1_DEVICE (cuda/cpu/auto)
        3. auto-detect (cuda si está disponible, cpu si no)
    """

    def __init__(self, model_id: str, device: str | None = None,
                 task_label: str = "T1"):
        """
        task_label: solo para logs (e.g. "T1" o "T2"). La clase es agnóstica
        a la tarea (cualquier AutoModelForSequenceClassification), pero el
        log muestra la etiqueta correcta para que sea trazable cuando T2
        reutiliza esta clase via get_t2_classifier().
        """
        self.task_label = task_label
        model_path = MODELS_DIR / model_id
        _verify_weights(model_path)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ModelNotAvailableError(
                "transformers/torch no instalados. "
                "Descomentar en requirements.txt: transformers>=4.45.0 y torch>=2.4.0"
            )

        self._torch = torch

        # Resolucion del device
        if device is None:
            device = os.getenv("T1_DEVICE", "auto").lower().strip()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            log.warning(f"{task_label}_DEVICE=cuda solicitado pero CUDA no disponible. Usando CPU.")
            device = "cpu"
        self.device = device

        log.info(f"Cargando modelo {task_label} '{model_id}' desde {model_path} (device={device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.eval()
        self.model.to(device)

        # Half precision en GPU acelera ~1.5x sin perdida apreciable de precision
        if device == "cuda":
            try:
                self.model = self.model.half()
                self._use_fp16 = True
                log.info(f"Modelo {task_label} '{model_id}' cargado en GPU con FP16")
            except Exception as e:
                log.warning(f"No se pudo convertir a FP16: {e}. Usando FP32.")
                self._use_fp16 = False
        else:
            self._use_fp16 = False
            log.info(f"Modelo {task_label} '{model_id}' cargado en {device} (FP32)")

        self.id2label = _read_id2label(model_path)
        self.max_length = 512
        self.model_id = model_id

    def warmup(self) -> None:
        """
        Ejecuta una pasada dummy para inicializar kernels CUDA / cache de allocator.
        Recomendado llamarlo al startup del servidor: la primera prediccion real
        del usuario no paga el costo de inicializacion.
        """
        try:
            log.info(f"Warming up {self.task_label} '{self.model_id}'...")
            t0 = __import__("time").perf_counter()
            _ = self.classify_single("Warmup dummy text for kernel init.")
            elapsed = __import__("time").perf_counter() - t0
            log.info(f"Warmup completed in {elapsed*1000:.0f}ms")
        except Exception as e:
            log.warning(f"Warmup failed (non-fatal): {e}")

    def _logits_to_result(self, probs_row: list[float]) -> dict[str, Any]:
        """Convierte una fila de probs en el dict de salida estándar."""
        all_probs = {self.id2label[i]: float(p) for i, p in enumerate(probs_row)}
        sorted_pairs = sorted(all_probs.items(), key=lambda kv: -kv[1])
        best_label, best_score = sorted_pairs[0]
        alternatives = [
            {"label": lbl, "score": round(score, 4)}
            for lbl, score in sorted_pairs[1:4]
        ]
        return {
            "label": best_label,
            "confidence": round(best_score, 4),
            "alternatives": alternatives,
            "all_probs": {k: round(v, 4) for k, v in all_probs.items()},
        }

    def classify_single(self, text: str) -> dict[str, Any]:
        """Clasifica un solo texto. Compatibilidad con uso anterior."""
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = self._torch.softmax(logits.float(), dim=-1).cpu().tolist()
        return self._logits_to_result(probs)

    def __call__(self, text: str) -> dict[str, Any]:
        """Hace al clasificador callable. Equivalente a classify_single."""
        return self.classify_single(text)

    def classify_batch(
        self, texts: list[str], micro_batch_size: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Clasifica una lista de textos en micro-batches.

        El micro_batch_size por defecto se elige según el device:
        - GPU: 32 (ajustar a la baja si VRAM < 8GB)
        - CPU: 8  (más allá de eso, el overhead Python no compensa)

        Override con variable de entorno T1_BATCH_SIZE.
        """
        if not texts:
            return []

        if micro_batch_size is None:
            env_bs = os.getenv("T1_BATCH_SIZE")
            if env_bs:
                micro_batch_size = int(env_bs)
            else:
                micro_batch_size = 32 if self.device == "cuda" else 8

        results: list[dict[str, Any]] = []
        for i in range(0, len(texts), micro_batch_size):
            chunk = texts[i:i + micro_batch_size]
            inputs = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                logits = self.model(**inputs).logits
                # softmax en float32 para precisión numérica
                probs = self._torch.softmax(logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                results.append(self._logits_to_result(row))
        return results


@lru_cache(maxsize=2)
def get_t1_classifier(model_id: str) -> "T1Classifier":
    """
    Carga (y cachea) el modelo T1 fine-tuned.
    Devuelve el wrapper T1Classifier que expone classify_single + classify_batch.

    Compatibilidad: para llamadas tipo `clf(text)`, el wrapper también es
    callable (delega a classify_single). Así no rompe código existente que
    asumía un Callable[[str], dict].
    """
    try:
        clf = T1Classifier(model_id)
        return clf
    except ModelNotAvailableError:
        raise
    except Exception as e:
        raise ModelNotAvailableError(f"Error al cargar T1 '{model_id}': {e}")


# ── T2: reutiliza T1Classifier (misma arquitectura RobertaForSequenceClassification) ─
#
# Decisión de diseño: el notebook de entrenamiento T2 (entrenamiento_t2.ipynb,
# celdas 12-14) confirma que el modelo recibe TEXTO CRUDO sin preprocessing,
# sin prefijos ni metadatos. La arquitectura es idéntica a T1 (RoBERTa-large +
# head de clasificación), y la lógica de inferencia es la misma:
#   tokenize(max_length=512, truncation=True, padding=longest) → forward → softmax.
# Esto hace innecesario duplicar la clase. Reutilizamos T1Classifier, que ya
# es agnóstica al número de labels (lee id2label del config.json: {0:NO_CONTR,
# 1:ES_CONTR}).
#
# Cache separado (lru_cache propio) para que T1 y T2 no se desalojen mutuamente.
@lru_cache(maxsize=2)
def get_t2_classifier(model_id: str) -> "T1Classifier":
    """
    Carga (y cachea) el modelo T2 fine-tuned para detección binaria de
    contribuciones. Devuelve un T1Classifier (la clase es agnóstica al número
    de labels — funciona con cualquier modelo de AutoModelForSequenceClassification).

    El device, FP16, batching y max_length se controlan con las mismas variables
    de entorno que T1 (T1_DEVICE, T1_BATCH_SIZE) por defecto. Si querés
    parámetros distintos para T2, definí T2_DEVICE / T2_BATCH_SIZE — tienen
    prioridad sobre los de T1 cuando este loader los lee.
    """
    # Las env vars T2_* sobrescriben las T1_* temporalmente durante la
    # construcción del clasificador. Esto permite, por ej., usar GPU para T1
    # y CPU para T2 en máquinas con poca VRAM.
    t2_device = os.getenv("T2_DEVICE")
    t2_batch_size = os.getenv("T2_BATCH_SIZE")
    _saved = {}
    if t2_device is not None:
        _saved["T1_DEVICE"] = os.environ.get("T1_DEVICE")
        os.environ["T1_DEVICE"] = t2_device
    if t2_batch_size is not None:
        _saved["T1_BATCH_SIZE"] = os.environ.get("T1_BATCH_SIZE")
        os.environ["T1_BATCH_SIZE"] = t2_batch_size
    try:
        clf = T1Classifier(model_id, task_label="T2")
        return clf
    except ModelNotAvailableError:
        raise
    except Exception as e:
        raise ModelNotAvailableError(f"Error al cargar T2 '{model_id}': {e}")
    finally:
        # Restaurar env vars al estado previo
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def list_available_models() -> list[str]:
    """Lista los modelos con pesos REALES descargados en MODELS_DIR."""
    if not MODELS_DIR.exists():
        return []
    return [
        d.name for d in MODELS_DIR.iterdir()
        if d.is_dir() and (
            list(d.glob("pytorch_model.bin"))
            or list(d.glob("pytorch_model-*-of-*.bin"))
            or list(d.glob("model.safetensors"))
            or list(d.glob("model-*-of-*.safetensors"))
        )
    ]


def get_model_metadata(model_id: str) -> dict[str, Any]:
    """
    Lee metadata del modelo desde config.json y eval_metrics.json (si existe).
    Útil para el endpoint /api/models/<id>/info.
    """
    model_path = MODELS_DIR / model_id
    if not (model_path / "config.json").exists():
        return {"model_id": model_id, "available": False}

    with open(model_path / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    meta = {
        "model_id": model_id,
        "available": True,
        "architecture": config.get("architectures", ["unknown"])[0],
        "base_model": config.get("_name_or_path", "unknown"),
        "num_labels": config.get("num_labels", len(config.get("id2label", {}))),
        "labels": list(config.get("label2id", {}).keys()),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "hidden_size": config.get("hidden_size"),
    }

    # Si el equipo de modelos guardó eval_metrics.json, exponerlo
    eval_path = model_path / "eval_metrics.json"
    if eval_path.exists():
        try:
            with open(eval_path, encoding="utf-8") as f:
                meta["eval_metrics"] = json.load(f)
        except Exception as e:
            log.warning(f"No se pudo leer eval_metrics.json: {e}")

    return meta
