# MAIA · Proyecto Despliegue Soluciones - Tarea 1 + Tarea 2

Análisis retórico automático y detección de contribuciones científicas en artículos académicos en español.

---

## Inicio rápido

### Windows

```cmd
start.bat
```

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

Abrir en el navegador: **http://localhost:8000**
Documentación API: **http://localhost:8000/docs**

---

## Estructura del proyecto

```
maia_proyecto_grado_g2/
│
├── backend/
│   ├── main.py              ← FastAPI: API + servidor de archivos estáticos
│   ├── model_loader.py      ← Carga perezosa de modelos fine-tuned (A5)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── index.html           ← Estructura HTML (sin lógica inline)
│   ├── css/
│   │   └── styles.css       ← Estilos: dark/light, layout, modal, componentes
│   ├── js/
│   │   └── app.js           ← Lógica: heurística, render, PDF, exportación
│   └── assets/
│       └── logo.jpg         ← Logo MAIA
│
├── models/                  ← (vacío) Aquí van los pesos descargados
│   └── README.md            ← Instrucciones de descarga de modelos
│
├── start.bat                ← Arranque Windows (un doble clic)
├── start.sh                 ← Arranque Linux/macOS
├── .env.example             ← Variables de entorno (copiar a .env)
├── .gitignore
├── docker-compose.yml
└── README.md
```

---

## Instalación manual (sin los scripts de arranque)

```bash
# 1. Crear entorno virtual
python -m venv venv

# Windows:
venv\Scripts\activate.bat

# Linux/macOS:
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Arrancar
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Modelos (~1.5 GiB por modelo)

Los pesos no se incluyen en este ZIP por su tamaño.

### Estructura esperada

```
models/
├── scibert-es/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── pytorch_model.bin       ← ~650 MB
├── beto-uncased/
│   └── ...
└── mdeberta-v3/
    └── ...
```

### Descarga

```bash
# Opción A: HuggingFace Hub (cuando el modelo esté publicado)
pip install huggingface-hub
huggingface-cli download maia-uniandes/scibert-es --local-dir models/scibert-es

# Opción B: enlace directo (Google Drive / OneDrive)
# Ver instrucciones en models/README.md
```

### Activar un modelo descargado

1. Coloca los archivos en `models/<model-id>/`
2. En `backend/main.py` busca el modelo en `AVAILABLE_MODELS` y cambia `"available": False` → `"available": True`
3. Reinicia el servidor

---

## Variables de entorno

```bash
# Copiar el ejemplo y editar
cp .env.example .env
```

| Variable | Descripción | Default |
|---|---|---|
| `PORT` | Puerto del servidor | `8000` |
| `HOST` | Host de escucha | `0.0.0.0` |
| `MODELS_DIR` | Ruta a los pesos de modelos | `./models` |
| `OPENAI_API_KEY` | Para GPT-4o (A7) | vacío |
| `GOOGLE_API_KEY` | Para Gemini (A7) | vacío |

---

## Funcionalidades

### Tarea 1 — Segmentación retórica (8 etiquetas)

| Etiqueta | Color | Sección |
|---|---|---|
| INTRO | 🔵 Azul | Introducción |
| BACK | 🟣 Violeta | Antecedentes |
| METH | 🟡 Ámbar | Metodología |
| RES | 🟢 Verde | Resultados |
| DISC | 🩵 Cian | Discusión |
| CONTR | 🔴 Rojo | Contribución |
| LIM | 🟠 Naranja | Limitaciones |
| CONC | 🔷 Índigo | Conclusiones |

### Tarea 2 — Detección de contribuciones

- T1 y T2 son **independientes**: una contribución puede estar en cualquier sección retórica.
- Las oraciones con marcadores de aporte se **resaltan en rojo** dentro del párrafo.
- Puntuación de confianza (0–100%) para T1 y T2 en el tooltip de cada párrafo.

### Archivos soportados

| Formato | Detección de columnas | Detección de tamaño de hoja |
|---|---|---|
| PDF | Sí (1, 2 o 3 columnas) | Sí (Carta, A4, Oficio, Folio) |
| DOCX / DOC | No necesario (Mammoth) | Sí (desde XML interno) |
| TXT | — | Siempre Carta |

### Exportación de resultados

- **JSON** — todos los campos (etiqueta, confianza, texto)
- **CSV** — tabla plana para Excel / pandas
- **HTML** — informe visual standalone con colores y highlights

---

## Modelos planificados

| Actividad | Modelos | Estado |
|---|---|---|
| A5 — Encoders | SciBETO-large, BETO-uncased, mDeBERTa-v3 | Pendiente |
| A6 — LLM open | Llama 3.1 8B, Mistral 7B, Gemma 2 9B | Pendiente |
| A7 — LLM API | GPT-4o, Gemini 1.5 Pro | Pendiente |

El modo **heurístico** (JS local) siempre está disponible como fallback sin necesidad de modelos.

---


---

## Interpretación de los resultados

### Barra de métricas (tras el análisis)

```
Modelo: Heurístico (sin modelo)  Modo: heuristic  Párrafos: 8  Palabras: 276        ⏱ 3 ms
```

| Campo | Descripción |
|---|---|
| **Modelo** | Nombre del motor de análisis utilizado. *Heurístico (sin modelo)* indica que se usaron reglas léxicas locales sin modelo entrenado. |
| **Modo** | Familia del motor: `heuristic` (reglas JS), `encoder` (BERT/DeBERTa fine-tuned), `llm_open` (Llama/Mistral), `llm_api` (GPT-4o/Gemini). |
| **Párrafos** | Número de segmentos en los que el sistema dividió el texto. Se basa en dobles saltos de línea o en grupos de ~50 palabras si el texto no tiene párrafos marcados. |
| **Palabras** | Total de tokens (palabras separadas por espacio) en el texto analizado. |
| **⏱ tiempo** | Latencia de inferencia en ms. Heurístico < 20 ms · Encoders ~200–500 ms · LLM API depende de la red. |

### Distribución retórica — KPIs de Tarea 1

Tras el análisis aparece una fila de 8 tarjetas, una por categoría retórica:

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ...
│    1    │  │    1    │  │    0    │
│   13%   │  │   13%   │  │    0%   │
│ Intro.  │  │ Anteced.│  │ Discus. │
└─────────┘  └─────────┘  └─────────┘
```

- **Número (grande)** — párrafos clasificados en esa categoría.
- **Porcentaje** — proporción sobre el total de párrafos del texto:
  ```
  % = (párrafos en categoría / total párrafos) × 100
  ```
- Los porcentajes de las 8 categorías suman **100 %**.
- Una tarjeta con **0 / 0 %** indica que ese tipo retórico no fue detectado en el texto.
- La distribución esperada en un artículo científico completo: INTRO y BACK al inicio, METH y RES en el cuerpo, DISC/CONTR/LIM/CONC hacia el final.

## Docker

```bash
docker compose up --build
# http://localhost:8000
```

---

## Proyecto académico

Maestría en Inteligencia Artificial — Universidad de los Andes · 2026  

