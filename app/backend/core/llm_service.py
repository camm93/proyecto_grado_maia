"""
llm_service.py — Inferencia T1 y T2 con LLMs (Llama local + GPT API).

Soporta dos modos de inferencia local para Llama:
  - GPU (default si CUDA disponible): 4-bit quantization vía bitsandbytes,
    ~10 GB VRAM, latencia 1-3s por fragmento.
  - CPU (auto si no hay CUDA): FP16 sin quantization, ~16 GB RAM, latencia
    30-90s por fragmento. Útil para testing local (Windows/Mac sin GPU).
    NO recomendado para producción — la latencia hace inviable el demo.

GPT no necesita GPU — es API, solo requiere `OPENAI_API_KEY`.

T2 (detección de contribuciones) tiene 4 variantes: Llama ZS, Llama FS-8,
GPT ZS, GPT FS-8. T1 (segmentación retórica) tiene 2: Llama FS-14 y GPT
FS-14 (mejores métricas según notebook v2).

Diseño:
  - `LlamaService` y `GPTService` son singletons (lru_cache) para no
    recargar Llama o reabrir conexión OpenAI por cada request.
  - `run_t2_llm(model_id, fragments)` es el entry point — el caller
    (api/contributions.py) lo invoca con cualquiera de los 4 model_ids
    LLM y devuelve la lista de resultados con el mismo schema que SciBETO.

Variables de entorno:
  OPENAI_API_KEY    — credencial OpenAI (obligatoria para GPT)
  LLAMA_MODEL_NAME  — default 'meta-llama/Llama-3.1-8B-Instruct'
  HF_TOKEN          — para descargar Llama (modelo gated)
  T2_LLM_BATCH      — cantidad de fragments procesados en paralelo (default 1)
  T2_LLM_LOG        — 'true' para loggear cada respuesta cruda del LLM
"""

from __future__ import annotations
import os
import time
import logging
from functools import lru_cache
from typing import Any

from .prediction_logger import log_t2_prediction
from ..model_loader import ModelNotAvailableError

log = logging.getLogger("maia.t2_llm")

# ── Prompt idéntico al notebook v4 ────────────────────────────────────────────
SYSTEM_PROMPT_T2 = """Eres un anotador experto en literatura científica en español. \
Tu tarea es decidir si un fragmento de texto contiene una contribución científica \
explícita del autor del trabajo donde aparece el fragmento.

Una CONTRIBUCIÓN CIENTÍFICA es un aporte original que el autor presenta como \
suyo: un método novedoso propuesto, un hallazgo empírico relevante obtenido en \
este trabajo, un avance conceptual o una herramienta desarrollada por el autor.

NO son contribuciones:
- Descripciones de antecedentes o trabajos previos de otros autores
- Métodos estándar de la disciplina (a menos que el autor los modifique)
- Resultados de experimentos previos citados como referencia
- Observaciones generales sin relación directa con el aporte del trabajo

Responde ÚNICAMENTE con una palabra: "SI" o "NO". No agregues explicaciones."""

# ── Few-shot examples ────────────────────────────────────────────────────────
# Set k=8 curado a mano por el equipo (no por heurística automática).
# Balance 4 positivos + 4 negativos, ORDEN ALTERNADO (SI, NO, SI, NO, ...).
#
# Orden alternado: evita sesgo posicional hacia la última clase observada en
# el contexto del LLM. En few-shot prompting hay evidencia de que los últimos
# ejemplos pesan más; alternar neutraliza ese efecto.
#
# Reproducibilidad: el SHA256 del set se publica en release_manifest.json.
# El demo, el paper y el notebook v5 deben usar EXACTAMENTE estos textos para
# que las métricas reportadas coincidan con el comportamiento en producción.
#
# Cómo verificar consistencia:
#   from backend.core.llm_service import FEW_SHOT_EXAMPLES
#   import hashlib, json
#   canonical = json.dumps(
#       [{"label": lbl, "text": txt} for lbl, txt in FEW_SHOT_EXAMPLES],
#       ensure_ascii=False, sort_keys=True,
#   )
#   sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
#   # SHA256 esperado: 4ccc1238b0afcc0b3885d818d911881ce8f372d9c17ed6bc43b918d14e20d509

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # ── 1 (SI) — ML/grafos: arquitectura novedosa + benchmarks
    ("SI",
     "En este artículo presentamos Graphformer-V2, una arquitectura que integra "
     "atención local-global mediante un mecanismo de ventanas dinámicas y grafos "
     "adaptativos. A diferencia de los Graph Transformers anteriores, nuestro "
     "modelo reduce la complejidad cuadrática a lineal-logarítmica mediante un "
     "muestreo jerárquico de nodos. Logramos un nuevo estado del arte en ocho "
     "benchmarks de grafos (incluyendo ogbg-molpcba y PCQM4Mv2) con un 12% menos "
     "de parámetros y 47% menos FLOPs que el mejor modelo previo. Además, "
     "demostramos una mejor generalización en escenarios de distribución fuera "
     "del dominio."),

    # ── 2 (NO) — survey/revisión de literatura, sin aporte original
    ("NO",
     "Este trabajo revisa los principales avances en el campo del aprendizaje "
     "profundo aplicados a la detección de fraudes bancarios durante los últimos "
     "cinco años. Analizamos 34 artículos relevantes y comparamos el rendimiento "
     "de CNN, LSTM y modelos basados en Transformers. Los resultados confirman "
     "que los modelos más recientes obtienen mejor desempeño, aunque siguen "
     "existiendo desafíos en cuanto a interpretabilidad y datos desbalanceados. "
     "Se discuten posibles direcciones futuras."),

    # ── 3 (SI) — biomedicina/química: compuesto nuevo + eficacia preclínica
    ("SI",
     "Desarrollamos un nuevo inhibidor covalente (BTK-CI-23) altamente selectivo "
     "contra la tirosina quinasa de Bruton. El compuesto muestra una IC50 de "
     "0.8 nM y más de 500 veces de selectividad sobre otras quinasas TEC. En "
     "modelos de xenoinjerto de linfoma de células del manto, logró una regresión "
     "tumoral completa en el 83% de los casos con dosificación oral diaria. La "
     "novedad radica en el diseño de un pocket alostérico que reduce "
     "significativamente los efectos off-target observados en inhibidores de "
     "primera y segunda generación."),

    # ── 4 (NO) — fine-tuning estándar replicado, sin innovación
    ("NO",
     "Implementamos un modelo BERT fine-tuned para clasificar reseñas de "
     "productos en español. Utilizamos un dataset de 15.000 reseñas recolectadas "
     "de Mercado Libre. El modelo alcanzó una accuracy del 87.3% y un F1-score "
     "de 0.85. Estos resultados son consistentes con lo reportado en la "
     "literatura para tareas similares de análisis de sentimiento. Realizamos "
     "experimentos de ablación para validar la importancia del preprocesamiento "
     "de texto."),

    # ── 5 (SI) — multimodal LLMs: método novedoso + SOTA
    ("SI",
     "Proponemos un método de alineación multimodal (CrossModal-DPO) que alinea "
     "modelos de lenguaje con videos sin necesidad de anotaciones humanas. "
     "Mediante la optimización directa de preferencia en pares video-texto "
     "generados automáticamente, superamos en 9.4 puntos el estado del arte en "
     "tareas de VideoQA y recuperación texto-video en MSRVTT y ActivityNet. El "
     "modelo demuestra emergente razonamiento espacio-temporal sin entrenamiento "
     "supervisado explícito en estas capacidades."),

    # ── 6 (NO) — comparación de optimizadores conocidos, sin aporte nuevo
    ("NO",
     "En este estudio comparamos el rendimiento de cinco algoritmos de "
     "optimización (Adam, RMSprop, SGD, AdamW y Lion) en el entrenamiento de "
     "redes residuales ResNet-50 sobre ImageNet. Nuestros experimentos confirman "
     "que AdamW ofrece el mejor balance entre velocidad de convergencia y "
     "precisión final, resultados similares a los encontrados en trabajos "
     "previos. Se proporcionan recomendaciones prácticas para investigadores."),

    # ── 7 (SI) — biomedicina/clínica: primer estudio en humanos
    ("SI",
     "Presentamos evidencias de que la suplementación con urolitina A "
     "mitocondrial durante 16 semanas revierte significativamente el declive "
     "muscular en adultos mayores de 65 años. El ensayo doble ciego con 120 "
     "participantes mostró un aumento del 38% en fuerza de piernas y una mejora "
     "del 27% en rendimiento mitocondrial medido por biopsia. Este es el primer "
     "estudio en humanos que demuestra reversión de sarcopenia mediante "
     "modulación de mitofagia en población anciana."),

    # ── 8 (NO) — benchmark local con métodos estándar, sin novedad metodológica
    ("NO",
     "Aplicamos técnicas estándar de machine learning para predecir el precio "
     "de viviendas en Bogotá usando datos abiertos del distrito. El modelo "
     "XGBoost obtuvo un MAE de 18.4 millones de pesos. Este trabajo sirve como "
     "benchmark local y confirma que las variables de ubicación, área y estrato "
     "socioeconómico siguen siendo las más predictivas, en línea con la "
     "literatura internacional."),
]


def parse_llm_answer(text: str) -> int:
    """Extrae 1 (ES_CONTR) o 0 (NO) de la respuesta cruda del LLM."""
    if not isinstance(text, str):
        return 0
    txt = text.strip().lower()
    if txt.startswith("si") or txt.startswith("sí") or txt.startswith("yes"):
        return 1
    if txt.startswith("no"):
        return 0
    return 0  # default a NO ante respuesta ambigua


# ════════════════════════════════════════════════════════════════════════════
#  LLAMA SERVICE (local, 4-bit)
# ════════════════════════════════════════════════════════════════════════════

class LlamaService:
    """Singleton que mantiene el modelo Llama 3.1 8B cargado.

    Dos modos de carga:
      - GPU: 4-bit quantization vía bitsandbytes (~10 GB VRAM, latencia ~1-3s).
      - CPU: FP16 sin quantization (~16 GB RAM, latencia 30-90s/fragmento).
              Activado automáticamente si no hay CUDA. NO usa bitsandbytes
              (incompatible con CPU). Edwin lo pidió explícito para testing
              en Windows local sin GPU.
    """

    def __init__(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ModelNotAvailableError(
                f"Llama requiere transformers + torch: {e}. "
                f"Instalar con: pip install transformers torch"
            )

        self.torch = torch
        model_name = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise ModelNotAvailableError(
                "HF_TOKEN no configurado. Setearlo en .env o env var antes de "
                "usar Llama (modelo gated, requiere aceptar la licencia en "
                "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)."
            )

        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"

        if use_cuda:
            # ── Camino GPU: 4-bit quantization con bitsandbytes ─────────────
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ModelNotAvailableError(
                    "bitsandbytes no instalado (requerido para 4-bit GPU). "
                    "pip install bitsandbytes accelerate"
                )
            log.info(f"Cargando Llama {model_name} en GPU 4-bit (~10 GB VRAM)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, quantization_config=bnb_config,
                    device_map="auto", token=hf_token,
                )
                self.model.eval()
            except Exception as e:
                raise ModelNotAvailableError(
                    f"No se pudo cargar Llama en GPU: {e}. "
                    f"Verificar HF_TOKEN, acceso al repo gated, y VRAM disponible."
                )
            vram_gb = torch.cuda.memory_allocated() / 1e9
            log.info(f"[OK] Llama cargado en GPU 4-bit. VRAM: {vram_gb:.2f} GB")

        else:
            # ── Camino CPU: FP16 sin quantization ──────────────────────────
            # Llama 8B en FP16 ocupa ~16 GB de RAM. La primera carga toma
            # 5-15 min en Windows (descarga de HF + load). Cada inferencia
            # tarda 30-90s en CPU típico. Edwin lo aceptó explícitamente
            # para testing local.
            log.warning(
                "CUDA no disponible. Cargando Llama en CPU FP16 (~16 GB RAM, "
                "30-90s por inferencia). La primera carga puede tomar 5-15 min "
                "(descarga ~16 GB desde HF + load). Para producción usar GPU."
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # 16 GB en vez de 32
                    low_cpu_mem_usage=True,     # carga sharded para evitar OOM
                    token=hf_token,
                )
                # Forzar device CPU (sin device_map="auto" que necesita accelerate)
                self.model = self.model.to("cpu")
                self.model.eval()
            except MemoryError as e:
                raise ModelNotAvailableError(
                    f"OOM cargando Llama en CPU FP16 (~16 GB RAM requeridos): {e}. "
                    f"Tu maquina no tiene suficiente memoria. Recomendado: "
                    f"usar 'gpt-4o-mini-t*' (API, sin RAM local) o esperar al "
                    f"deploy en EC2 g4dn.xlarge."
                )
            except Exception as e:
                raise ModelNotAvailableError(
                    f"No se pudo cargar Llama en CPU: {e}. "
                    f"Verificar HF_TOKEN, conexion a HF Hub, y >=16 GB RAM libres."
                )
            log.info("[OK] Llama cargado en CPU FP16. Inferencia sera lenta.")

    def _build_prompt(self, text: str, few_shot: bool) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT_T2}]
        if few_shot:
            for lbl, ex_text in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user",
                                 "content": f"Fragmento:\n{ex_text}\n\nRespuesta:"})
                messages.append({"role": "assistant", "content": lbl})
        messages.append({"role": "user",
                         "content": f"Fragmento:\n{text}\n\nRespuesta:"})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def predict(self, text: str, few_shot: bool = False) -> tuple[int, str, float]:
        """Devuelve (pred 0/1, raw_answer, latency_sec)."""
        prompt = self._build_prompt(text, few_shot)
        # k=8 con ejemplos curados largos consume ~1400 tokens de prompt + texto.
        # Con max_length=4096 quedan ~2600 tokens para el fragmento de query,
        # lo que cubre fragmentos de hasta ~10k chars sin truncar.
        # Llama 3.1 soporta hasta 131k tokens de contexto, no hay límite duro.
        max_len = 4096 if few_shot else 2048
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len,
        ).to(self.model.device)

        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_llm_answer(answer), answer, elapsed


@lru_cache(maxsize=1)
def get_llama_service() -> LlamaService:
    return LlamaService()


# ════════════════════════════════════════════════════════════════════════════
#  GPT SERVICE (API)
# ════════════════════════════════════════════════════════════════════════════

class GPTService:
    """Cliente OpenAI con cache."""

    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ModelNotAvailableError(f"openai package no instalado: {e}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ModelNotAvailableError(
                "OPENAI_API_KEY no configurada. Setear env var antes de usar GPT."
            )

        self.client = OpenAI(api_key=api_key)
        self.model_name = os.getenv("GPT_MODEL_NAME", "gpt-4o-mini")
        log.info(f"✓ GPT service inicializado con model={self.model_name}")

    def predict(self, text: str, few_shot: bool = False,
                max_retries: int = 3) -> tuple[int, str, float]:
        """Devuelve (pred 0/1, raw_answer, latency_sec). Reintenta hasta 3 veces."""
        text_truncated = text[:4000] if len(text) > 4000 else text

        messages = [{"role": "system", "content": SYSTEM_PROMPT_T2}]
        if few_shot:
            for lbl, ex_text in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user",
                                 "content": f"Fragmento:\n{ex_text}\n\nRespuesta:"})
                messages.append({"role": "assistant", "content": lbl})
        messages.append({"role": "user",
                         "content": f"Fragmento:\n{text_truncated}\n\nRespuesta:"})

        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0, max_tokens=8, timeout=30,
                )
                elapsed = time.perf_counter() - t0
                answer = resp.choices[0].message.content.strip()
                return parse_llm_answer(answer), answer, elapsed
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    log.error(f"GPT inferencia falló tras {max_retries} intentos: {e}")
                    return 0, "ERROR", 0.0


@lru_cache(maxsize=1)
def get_gpt_service() -> GPTService:
    return GPTService()


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — run_t2_llm()
# ════════════════════════════════════════════════════════════════════════════

# Mapeo model_id → (servicio, few_shot)
LLM_DISPATCH = {
    "llama-3.1-8b-t2":      ("llama", False),
    "llama-3.1-8b-t2-fs8":  ("llama", True),
    "gpt-4o-mini-t2":       ("gpt",   False),
    "gpt-4o-mini-t2-fs8":   ("gpt",   True),
}


LABEL_POSITIVE = "ES_CONTRIBUCION"
LABEL_NEGATIVE = "NO_ES_CONTRIBUCION"


def _build_rhetorical_context(is_contribution: bool, frag: dict) -> str:
    if not is_contribution:
        return ""
    t1_name = frag.get("t1_label_name", frag.get("t1_label", ""))
    t1_conf = frag.get("t1_confidence", 0)
    zone = frag.get("rhetorical_zone", "")
    return (f"Contribución detectada en sección '{t1_name}' "
            f"({t1_conf*100:.0f}% conf. retórica) · {zone}")


def run_t2_llm(model_id: str, fragments: list[dict]) -> list[dict]:
    """
    Entry point unificado para los 4 modelos LLM.

    model_id ∈ {
        'llama-3.1-8b-t2',      # Llama zero-shot
        'llama-3.1-8b-t2-fs4',  # Llama few-shot k=4
        'gpt-4o-mini-t2',       # GPT zero-shot
        'gpt-4o-mini-t2-fs4',   # GPT few-shot k=4
    }

    Schema de salida idéntico a run_t2_scibeto / run_t2 heurístico.
    Lanza ModelNotAvailableError si el modelo no está configurado/disponible.
    """
    if model_id not in LLM_DISPATCH:
        raise ModelNotAvailableError(f"model_id LLM desconocido: {model_id}")

    backend, few_shot = LLM_DISPATCH[model_id]

    # Obtener el servicio (puede tirar ModelNotAvailableError si falta config)
    if backend == "llama":
        service = get_llama_service()
    else:  # gpt
        service = get_gpt_service()

    log.info(
        f"T2 LLM inferencia: {len(fragments)} fragments · "
        f"backend={backend} · few_shot={few_shot}"
    )

    results = []
    for idx, frag in enumerate(fragments):
        text = str(frag.get("text", ""))
        try:
            pred, raw_answer, elapsed = service.predict(text, few_shot=few_shot)
        except Exception as e:
            log.exception(f"LLM predict falló en fragment {idx}: {e}")
            pred, raw_answer, elapsed = 0, "ERROR", 0.0

        is_c = pred == 1
        label = LABEL_POSITIVE if is_c else LABEL_NEGATIVE
        # LLMs no devuelven probabilidades fácilmente con greedy → confidence
        # se reporta como 1.0 si dijo SI/NO con seguridad, 0.5 si fue ERROR.
        confidence = 0.5 if raw_answer == "ERROR" else 1.0

        # Log opcional (T2_LLM_LOG=true)
        if os.getenv("T2_LLM_LOG", "false").lower() == "true":
            log_t2_prediction(
                model_id=model_id,
                input_text=text,
                t1_context={
                    "t1_label": frag.get("t1_label", ""),
                    "rhetorical_zone": frag.get("rhetorical_zone", ""),
                    "relative_pos": frag.get("relative_pos"),
                },
                prediction={
                    "label": label, "confidence": confidence,
                    "raw_answer": raw_answer,
                    "all_probs": {},
                },
                elapsed_ms=int(elapsed * 1000),
            )

        results.append({
            "index": frag.get("index", idx),
            "is_contribution": is_c,
            "confidence": confidence,
            "label": label,
            "t1_label": frag.get("t1_label", ""),
            "t1_label_name": frag.get("t1_label_name", ""),
            "t1_color": frag.get("t1_color", "#9CA3AF"),
            "t1_confidence": frag.get("t1_confidence", 0),
            "rhetorical_zone": frag.get("rhetorical_zone", ""),
            "rhetorical_context": _build_rhetorical_context(is_c, frag),
            "char_start": frag.get("char_start"),
            "char_end": frag.get("char_end"),
            "word_count": frag.get("word_count"),
            "relative_pos": frag.get("relative_pos"),
            "_llm_raw_answer": raw_answer,  # útil para debugging
            "_llm_latency_sec": elapsed,
        })

    return results


# ════════════════════════════════════════════════════════════════════════════
# TAREA 1 — Segmentación retórica via LLM (Llama + GPT)
# ════════════════════════════════════════════════════════════════════════════
# Implementación idéntica al notebook task1_edwin_v2.ipynb (mejores métricas).
# Llama 3.1 8B fs14 → F1 macro 0.322; GPT-4o-mini fs14+JSON → F1 macro 0.416
# Ambos usan el MISMO few-shot block (single source of truth).
# ════════════════════════════════════════════════════════════════════════════

import json as _json
import re as _re

# Las 7 clases retóricas (sin CONTR — esa es de T2)
T1_LABELS = ["INTRO", "BACK", "METH", "RES", "DISC", "LIM", "CONC"]

# ── 14 ejemplos few-shot: 2 por clase × 7 clases ─────────────────────────────
# Single source of truth — idénticos a `_FEW_SHOT_RAW` del notebook v2 §7.
# SHA256 (calcular con _few_shot_sha_t1()) se publica en el manifest para
# auditoría bit-a-bit.
FEW_SHOT_EXAMPLES_T1: list[tuple[str, str]] = [
    ("INTRO", "Este estudio busca evaluar el impacto de las redes neuronales en la detección de fallas en sistemas industriales."),
    ("INTRO", "El presente trabajo se enmarca en el contexto de la creciente demanda de soluciones automatizadas para la clasificación documental."),
    ("BACK",  "Investigaciones previas de Smith et al. (2022) sugirieron que el litio es inestable en estas condiciones de presión."),
    ("BACK",  "Diversos autores han abordado el problema desde el enfoque de aprendizaje supervisado, destacando los trabajos de Pérez (2019) y García (2021)."),
    ("METH",  "Se utilizó un diseño experimental de bloques al azar con tres repeticiones por tratamiento y análisis ANOVA de dos vías."),
    ("METH",  "Para el procesamiento se aplicó un pipeline de tokenización, lematización con spaCy y vectorización TF-IDF previa al modelado."),
    ("RES",   "La tasa de error disminuyó significativamente de 0.85 a 0.12 tras la optimización del modelo (p<0.01)."),
    ("RES",   "Se observó una correlación positiva (r=0.78) entre las variables analizadas en los 245 sujetos del estudio."),
    ("DISC",  "Estos hallazgos sugieren que la presión atmosférica influye más de lo esperado, lo cual contradice estudios anteriores."),
    ("DISC",  "La interpretación de estos resultados debe considerar el contexto socioeconómico de la población muestreada."),
    ("LIM",   "Debido al tamaño reducido de la muestra, los resultados no pueden generalizarse a toda la población latinoamericana."),
    ("LIM",   "Una restricción importante fue la imposibilidad de acceder a datos longitudinales superiores a tres años."),
    ("CONC",  "En conclusión, el sistema propuesto es viable para implementaciones industriales a gran escala con margen de error aceptable."),
    ("CONC",  "Finalmente, se recomienda continuar investigando el efecto a largo plazo en estudios con muestras más amplias."),
]


def _build_few_shot_block_t1(examples=FEW_SHOT_EXAMPLES_T1) -> str:
    """Construye el bloque few-shot. Orden fijo (no shuffle) para que el SHA
    sea reproducible. En el notebook v2 hay shuffle con seed=42; acá lo
    omitimos porque el orden no afecta el desempeño con 14 ejemplos."""
    block = "Aquí tienes ejemplos de clasificación correcta:\n"
    for lbl, txt in examples:
        block += f'\nFragmento: "{txt}"\nCategoría: {lbl}\n'
    return block


FEW_SHOT_BLOCK_T1 = _build_few_shot_block_t1()


# Descripciones por clase para el system prompt T1 — idénticas al notebook v2 §7
_T1_CAT_DESCRIPTIONS = {
    "INTRO": "INTRO: Introducción, motivación, objetivos del estudio.",
    "BACK":  "BACK: Antecedentes, trabajos previos, estado del arte.",
    "METH":  "METH: Metodología, diseño experimental, materiales, procedimientos.",
    "RES":   "RES: Resultados obtenidos empíricamente (sin interpretación).",
    "DISC":  "DISC: Discusión, interpretación de resultados, implicaciones.",
    "LIM":   "LIM: Limitaciones del estudio, restricciones, sesgos.",
    "CONC":  "CONC: Conclusiones y trabajo futuro.",
}

SYSTEM_PROMPT_T1 = (
    "Eres un asistente experto en analizar artículos científicos en español.\n"
    f"Tu tarea es clasificar el fragmento de texto en EXACTAMENTE UNA de estas "
    f"{len(T1_LABELS)} categorías:\n"
    + "\n".join(f"- {_T1_CAT_DESCRIPTIONS[l]}" for l in T1_LABELS) + "\n\n"
    "Nota: El texto contiene palabras reemplazadas por \"[MASK]\". Deduce la "
    "categoría basándote en la semántica del resto del texto.\n\n"
    "Reglas: Responde ÚNICAMENTE con la sigla (ejemplo: INTRO, METH, DISC). "
    "Sin explicaciones."
)


# ── Parser tolerante T1 ──────────────────────────────────────────────────────
# Acepta tanto la sigla canónica ("INTRO") como variantes en español
# ("Introducción") y JSON ({"category": "INTRO"}). Cubre los modos de salida
# de Llama (raw) y GPT (JSON mode). Idéntico al notebook v2 §5.
_LLM_LABEL_MAP_T1 = {
    "INTRO": "INTRO", "BACK": "BACK", "METH": "METH", "RES": "RES",
    "DISC": "DISC", "LIM": "LIM", "CONC": "CONC", "CONTR": "CONTR",
    "INTRODUCCION": "INTRO", "INTRODUCCIÓN": "INTRO",
    "ANTECEDENTES": "BACK", "BACKGROUND": "BACK", "ESTADO DEL ARTE": "BACK",
    "METODOLOGIA": "METH", "METODOLOGÍA": "METH",
    "METODO": "METH", "MÉTODO": "METH",
    "RESULTADOS": "RES", "HALLAZGOS": "RES",
    "DISCUSION": "DISC", "DISCUSIÓN": "DISC",
    "LIMITACIONES": "LIM", "LIMITACION": "LIM",
    "CONCLUSION": "CONC", "CONCLUSIÓN": "CONC", "CONCLUSIONES": "CONC",
    "CONTRIBUCION": "CONTR", "CONTRIBUCIÓN": "CONTR", "CONTRIBUCIONES": "CONTR",
}


def parse_llm_label_t1(raw_response: str) -> str:
    """Convierte la respuesta del LLM a una sigla canónica de T1_LABELS o
    'UNKNOWN' si no se reconoce. Tolerante a JSON, variantes en español y
    mayúsculas/minúsculas."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        return "UNKNOWN"
    cleaned = raw_response.upper().strip()

    # Intento 1: JSON mode (GPT)
    if cleaned.startswith("{"):
        try:
            obj = _json.loads(raw_response)
            cat = str(obj.get("category", "")).upper().strip()
            if cat in _LLM_LABEL_MAP_T1:
                canonical = _LLM_LABEL_MAP_T1[cat]
                return canonical if canonical in T1_LABELS else "UNKNOWN"
        except (_json.JSONDecodeError, AttributeError):
            pass

    # Intento 2: búsqueda con \b — ordenado por longitud descendente para
    # que "INTRODUCCIÓN" matchee antes que "INTRO"
    for needle in sorted(_LLM_LABEL_MAP_T1.keys(), key=len, reverse=True):
        if _re.search(rf"\b{_re.escape(needle)}\b", cleaned):
            canonical = _LLM_LABEL_MAP_T1[needle]
            return canonical if canonical in T1_LABELS else "UNKNOWN"

    return "UNKNOWN"


# ── Métodos predict_t1 inyectados en las clases existentes ───────────────────
def _llama_predict_t1(self, snippet: str, few_shot: bool = True) -> tuple[str, str, float]:
    """Devuelve (label_canónica, raw_response, elapsed_sec) para un snippet T1.

    `few_shot=True` (default): incluye el bloque FEW_SHOT_BLOCK_T1 (k=14, 2 por clase).
    `few_shot=False`: zero-shot puro, solo SYSTEM_PROMPT_T1. Variante agregada
    en v3 tras el notebook task1_edwin_v3 que evaluó ZS vs FS para los 2 LLMs.
    """
    system_content = SYSTEM_PROMPT_T1
    if few_shot:
        system_content = system_content + "\n\n" + FEW_SHOT_BLOCK_T1
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": f'Fragmento: "{snippet}"\nCategoría:'},
    ]
    prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # max_length=7000 (FS-14) deja margen para snippet de 1000 palabras
    # (~1700 tokens) + system prompt (~200 tokens) + 14 few-shot (~1400 tokens).
    # En zero-shot se baja a 3000 (no necesita los 1400 tokens del bloque FS).
    max_len = 7000 if few_shot else 3000
    inputs = self.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_len,
    ).to(self.model.device)

    t0 = time.perf_counter()
    with self.torch.no_grad():
        out = self.model.generate(
            **inputs, max_new_tokens=8,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_llm_label_t1(answer), answer, elapsed


def _gpt_predict_t1(self, snippet: str, few_shot: bool = True) -> tuple[str, str, float]:
    """Devuelve (label_canónica, raw_response, elapsed_sec) para un snippet T1.
    Usa JSON mode + retry exponencial (vía la lib openai).

    `few_shot=True` (default): incluye el bloque FEW_SHOT_BLOCK_T1 (k=14).
    `few_shot=False`: zero-shot. En el notebook v3, GPT-ZS supera a GPT-FS
    en T1 (F1 0.447 vs 0.414) — el ZS es ahora el modelo recomendado.
    """
    system_content = SYSTEM_PROMPT_T1
    if few_shot:
        system_content = system_content + "\n\n" + FEW_SHOT_BLOCK_T1
    system_content = system_content + '\n\nResponde en formato JSON: {"category": "SIGLA"}'
    t0 = time.perf_counter()
    try:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f'Fragmento: "{snippet}"'},
            ],
            temperature=0.0, max_tokens=16, timeout=30,
            response_format={"type": "json_object"},
        )
        elapsed = time.perf_counter() - t0
        answer = resp.choices[0].message.content
        return parse_llm_label_t1(answer), answer, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log.warning(f"GPT predict_t1 error: {type(e).__name__}: {e}")
        return "UNKNOWN", f"ERROR: {e}", elapsed


# Inyección dinámica de los métodos (mantengo las clases definidas arriba)
LlamaService.predict_t1 = _llama_predict_t1
GPTService.predict_t1 = _gpt_predict_t1


# ── Dispatch T1 ──────────────────────────────────────────────────────────────
# (backend, few_shot) — few_shot=True usa FEW_SHOT_BLOCK_T1 (k=14, 2 por clase),
# few_shot=False usa solo SYSTEM_PROMPT_T1. Variantes ZS agregadas en v3 tras el
# notebook task1_edwin_v3 que evaluó ambas configuraciones para Llama y GPT.
LLM_DISPATCH_T1 = {
    "llama-3.1-8b-t1":      ("llama", True),    # few-shot k=14 (F1 0.322)
    "llama-3.1-8b-t1-zs":   ("llama", False),   # zero-shot   (F1 0.234)
    "gpt-4o-mini-t1":       ("gpt",   True),    # few-shot k=14 (F1 0.414)
    "gpt-4o-mini-t1-zs":    ("gpt",   False),   # zero-shot   (F1 0.447) - mejor T1
}


def run_t1_llm(model_id: str, paragraphs: list[str]) -> list[dict]:
    """Clasifica una lista de párrafos pre-segmentados con un LLM.

    A diferencia de run_t2_llm (que recibe `fragments` con metadata),
    este recibe párrafos crudos (strings) porque el segmentado ya lo hizo
    el caller (api/segment.py via split_paragraphs).

    Devuelve list[dict] con el schema T1: cada item tiene `label`,
    `confidence`, `raw_answer`. El caller arma el output final con
    metadata posicional.
    """
    if model_id not in LLM_DISPATCH_T1:
        raise ModelNotAvailableError(f"model_id LLM T1 desconocido: {model_id}")

    backend, few_shot = LLM_DISPATCH_T1[model_id]
    service = get_llama_service() if backend == "llama" else get_gpt_service()

    log.info(f"T1 LLM inferencia: {len(paragraphs)} parrafos · "
             f"backend={backend} · few_shot={few_shot}")
    out = []
    for snippet in paragraphs:
        label, raw, elapsed = service.predict_t1(snippet, few_shot=few_shot)
        # Confidence: LLMs greedy no exponen probs. 1.0 si reconocido, 0.5 si UNK
        confidence = 0.5 if label == "UNKNOWN" else 1.0
        out.append({"label": label, "confidence": confidence,
                    "raw_answer": raw, "elapsed_sec": elapsed})
    return out
