# Guía de instalación — MAIA (para evaluadores)

**Proyecto:** MAIA — Segmentación Retórica de Documentos Científicos en Español y Detección de Contribuciones
**Equipo:** Grupo 2 · Maestría en Inteligencia Artificial · Universidad de los Andes · 2026
**Repositorio:** https://github.com/camm93/proyecto_grado_maia

---

## 1. Requisitos del entorno

### Software

| Componente | Versión mínima | Notas |
|---|---|---|
| Sistema operativo | Linux (Ubuntu 22.04 / Amazon Linux 2023 / RHEL 9) o Windows 10/11 | Para macOS los comandos son similares a Linux |
| Python | 3.10+ (recomendado 3.11) | Verificar con `python3 --version` |
| `git` | cualquier versión reciente | Para clonar el repositorio |
| `unzip` | cualquier versión | Solo Linux/Mac |
| Conexión a Internet | requerida en el primer arranque | Descarga ~22 GB de modelos |

### Hardware y espacio en disco

Dos perfiles posibles según los modelos que se quieran probar:

| Perfil | Disco libre | RAM | GPU |
|---|---|---|---|
| **Completo (con Llama 3.1 8B local)** | **30 GB** | 16 GB (32 GB si Llama corre en CPU) | Opcional — T4 16 GB recomendado |
| **Reducido (solo SciBETO + GPT API)** | **8 GB** | 8 GB | No requerida |

> **Nota práctica:** en la instancia de referencia del equipo (AWS EC2 g4dn.xlarge), `/data/app` ocupa **~29 GB** cuando todo está descargado y operativo (código + venv + 3 modelos SciBETO + cache de Llama 3.1 8B). Eso justifica los 30 GB del perfil completo.

### Credenciales opcionales

Para habilitar los modelos LLM hay que configurar dos claves en el archivo `.env` (se explica en el paso 4):

- **`OPENAI_API_KEY`** — para los modelos `gpt-4o-mini-*` (T1 y T2). El modelo `gpt-4o-mini-t1-zs` es el de mejor desempeño en Tarea 1 (F1 macro = 0.447).
- **`HF_TOKEN`** — para descargar Llama 3.1 8B desde Hugging Face. Es un modelo *gated*: requiere aceptar la licencia previamente en https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct y generar un token de acceso de lectura en https://huggingface.co/settings/tokens.

Si no se proveen, esas variantes quedan marcadas como **no disponibles** en el catálogo (○ gris en el selector); SciBETO y la heurística siguen funcionando sin ningún token.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/camm93/proyecto_grado_maia.git
cd proyecto_grado_maia
```

Estructura tras el clone:

```
proyecto_grado_maia/
├── INSTALACION.md       ← este documento
└── app/                 ← raíz del proyecto MAIA
    ├── backend/
    ├── frontend/
    ├── scripts/
    ├── tests/
    ├── docs/
    ├── notebooks/
    ├── release_manifest.json
    ├── .env.example
    ├── start.sh
    └── start.bat
```

---

## 3. Entrar a la carpeta del proyecto

```bash
cd app
```

A partir de aquí todos los comandos se ejecutan dentro de `proyecto_grado_maia/app/`.

---

## 4. Configurar variables de entorno

Copiar la plantilla y editar:

```bash
cp .env.example .env
```

Abrir `.env` con cualquier editor (`nano .env`, `vim .env`, Notepad, etc.) y completar **al menos** las dos claves opcionales si se quieren probar los LLMs:

```env
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

Los demás valores tienen defaults sensatos. Dos ajustes útiles según el hardware:

- **Sin GPU y con menos de 32 GB RAM** → comentar/poner en `false`:
  ```env
  LLAMA_PRELOAD=false
  ```
  Llama se cargará en *lazy mode* solamente cuando alguien lo seleccione (y consume ~16 GB RAM en CPU).
- **Para ignorar Llama por completo** → setear:
  ```env
  LLAMA_ENABLED=false
  ```

---

## 5. Ejecutar `start.sh`

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### Windows

```cmd
start.bat
```

El script realiza automáticamente:

1. **Verifica Python 3.10+** y avisa si falta.
2. **Crea el entorno virtual** (`./venv`) si no existe e instala las dependencias de `backend/requirements.txt` (~2 GB, una sola vez).
3. **Descarga los 3 modelos SciBETO** desde Google Drive con verificación SHA256 (~4 GB, ~5-8 min la primera vez).
4. **Levanta el backend FastAPI** en `http://127.0.0.1:8000` con uvicorn.
5. Si `HF_TOKEN` está configurado, **Llama 3.1 8B se descarga** desde Hugging Face en el primer arranque o en el primer request (depende de `LLAMA_PRELOAD`).

Esperar a ver en la consola:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 6. Verificar el deploy

### 6.1 Backend operativo

Desde otra terminal (o el browser):

```bash
curl http://127.0.0.1:8000/health
```

Respuesta esperada (resumen):

```json
{
  "status": "ok",
  "version": "2.1.0",
  "models_t1": ["heuristic", "scibeto-es-t1", "gpt-4o-mini-t1-zs", ...],
  "models_t2": ["heuristic", "scibeto-es-t2", ...]
}
```

### 6.2 Frontend

Abrir en el navegador: **http://127.0.0.1:8000**

Lo que se debe ver:

- Tema claro por defecto (con botón ☀️ para alternar a oscuro)
- Header con 4 botones: ☀️ · `?` · 📚 · 👥
- Dropdown de **Tarea 1** preseleccionado en `GPT-4o-mini (T1, zero-shot + JSON mode) -- mejor T1` (F1 macro 0.447)
- Dropdown de **Tarea 2** preseleccionado en `SciBETO-large text-only (T2) -- recomendado` (F1 macro 0.852)
- Botón **`Cargar ejemplo`** para probar con un texto de muestra

### 6.3 Probar el análisis

1. Hacer clic en **`Cargar ejemplo`** (carga un texto académico de prueba).
2. Hacer clic en **`▶ Analizar`**.
3. El panel derecho debe mostrar el texto segmentado en párrafos coloreados según su categoría retórica (INTRO, BACK, METH, RES, DISC, LIM, CONC) y resaltado en rojo donde detecte contribuciones.

---

## 7. Recursos adicionales

| Documento | Ubicación | Contenido |
|---|---|---|
| HANDOVER técnico | `app/docs/HANDOVER.md` | Estado completo del proyecto: métricas, hallazgos, arquitectura, pendientes |
| Material del paper | `app/docs/paper_material_v6_FINAL.md` | Resultados experimentales para el reporte |
| Notebook de evaluación T1 | `app/notebooks/task1_edwin_v3.ipynb` | Reproducibilidad: 5 modelos comparados (SciBETO + Llama ZS/FS + GPT ZS/FS) |
| API Docs (Swagger) | http://127.0.0.1:8000/docs | Documentación interactiva de los endpoints (`/api/segment`, `/api/contributions`, `/api/models`, `/api/few_shots`, etc.) |
| Botón 📚 en la app | dentro del frontend | Modal con los 14 ejemplos few-shot (T1) y 8 ejemplos (T2) publicados, con SHA256 para auditoría bit-a-bit |

---

## 8. Resolución de problemas comunes

### "Espacio insuficiente en disco"
Verificar con `df -h`. La carpeta donde se instala el proyecto necesita al menos 30 GB libres (perfil completo) u 8 GB (perfil reducido sin Llama).

### "ModuleNotFoundError: No module named 'torch'"
El script `start.sh` no terminó de instalar las dependencias. Activar manualmente el venv y reinstalar:
```bash
source venv/bin/activate    # Linux/Mac
pip install -r backend/requirements.txt
```

### Descarga de modelos falla con "SHA mismatch"
Borrar el cache y reintentar:
```bash
rm -rf models/.cache models/scibeto-es-t*
./start.sh
```

### Llama 3.1 8B se cuelga al cargar
Si la máquina tiene menos de 32 GB RAM y no tiene GPU, Llama no puede cargar en CPU FP16. Solución: editar `.env` y poner `LLAMA_ENABLED=false`. SciBETO y GPT siguen funcionando.

### Llave de OpenAI o HuggingFace inválida
Las variantes correspondientes quedan marcadas como `○` (no disponibles) en el dropdown. El sistema sigue operativo con los modelos disponibles; al seleccionar uno no disponible, el backend usa heurística como fallback y lo reporta en el banner superior del resultado.

---

## 9. Detener el servicio

En la terminal donde corre `start.sh`: presionar `Ctrl+C`.

Para eliminar completamente la instalación de la máquina (libera ~30 GB):

```bash
cd ..    # salir de app/
cd ..    # salir de proyecto_grado_maia/
rm -rf proyecto_grado_maia
```

---

## Contacto

Cualquier inconveniente durante la evaluación, contactar a cualquier integrante del equipo (los emails están dentro de la app, botón 👥 "Acerca del Proyecto").
