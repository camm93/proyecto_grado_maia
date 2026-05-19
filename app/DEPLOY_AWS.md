# Despliegue en AWS con GPU

Guía para correr el demostrador MAIA en una instancia EC2 con GPU NVIDIA.

## TL;DR

```bash
# 1. SSH a la instancia
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# 2. Subir el código (zip o git)
scp -i your-key.pem app.zip ubuntu@<ec2-public-ip>:~
ssh -i your-key.pem ubuntu@<ec2-public-ip>
unzip app.zip -d app && cd app

# 3. Levantar con docker compose (el entrypoint baja los pesos de Drive automáticamente)
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml logs -f maia   # seguir descarga ~5-8 min

# 4. Smoke test post-deploy
bash scripts/smoke_test.sh http://localhost:8000
```

Si el smoke test devuelve `0 fails`, el demo está listo.

---

## 1. Instancia recomendada

| Instancia | GPU | VRAM | Precio aprox. (us-east-1) | Notas |
|---|---|---|---|---|
| **g4dn.xlarge** | T4 | 16 GB | ~$0.53/h on-demand · ~$0.16/h spot | Recomendada para demo |
| g5.xlarge | A10G | 24 GB | ~$1.01/h | ~2× más rápida en latencia |
| p3.2xlarge | V100 | 16 GB | ~$3.06/h | Sobredimensionada |

Los 3 SciBETO-large (~1.3 GB cada uno en FP16) caben holgados en 16 GB con espacio para Llama 8B 4-bit si se activa.

**AMI:** *Deep Learning Base GPU AMI (Ubuntu 22.04)* — viene con drivers NVIDIA + `nvidia-container-toolkit` preinstalados.

**EBS:** mínimo 50 GB gp3 (el sistema + Docker + ~10 GB para pesos descomprimidos + ~4 GB de cache de ZIPs).

## 2. Setup en la instancia

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>

# Verificar GPU
nvidia-smi  # debe mostrar la T4 o A10G

# Verificar Docker + nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
# Si esto funciona, la GPU está disponible para Docker.

# Para el smoke test post-deploy
sudo apt install -y jq
```

## 3. Subir el código

```bash
# Opción A: zip pre-empaquetado (recomendado para defensa)
scp -i your-key.pem app.zip ubuntu@<ec2-public-ip>:~
ssh -i your-key.pem ubuntu@<ec2-public-ip>
unzip app.zip -d app && cd app

# Opción B: git clone (recomendado para CI/CD)
git clone https://github.com/camm93/proyecto_grado_maia.git
cd proyecto_grado_maia
```

## 4. Estrategia de pesos — Drive vs HF Hub vs S3

### Opción A — Google Drive vía `model_sync` (RECOMENDADA, default v5)

Los 3 ZIPs de pesos (~3.9 GB total) están publicados en Drive con permisos "Anyone with the link → Viewer". El módulo `backend/utils/model_sync.py` los descarga al primer arranque usando `gdown>=5.0` (que maneja la pantalla intermedia de virus-scan de Drive automáticamente) y verifica integridad SHA256 en dos niveles:

1. **Al descargar el ZIP:** SHA256 del archivo completo contra `release_manifest.json`. Si no coincide → re-descarga hasta 3 veces con backoff exponencial.
2. **Al arrancar el backend (cada vez):** SHA256 de cada archivo `required: true` extraído contra el manifest. Si falla → re-extrae del cache; si el cache también está roto → re-descarga.

No requiere `.env` ni credenciales. Solo arrancar.

```bash
# El entrypoint hace todo automáticamente al primer up:
docker compose -f docker-compose.gpu.yml up -d --build

# Seguir el progreso (1er arranque tarda ~5-8 min):
docker compose -f docker-compose.gpu.yml logs -f maia

# Esperá ver:
#   ═══ Sync de modelos via Google Drive (model_sync ensure_all) ═══
#   [scibeto-es-t1] Descarga intento 1/3 ...
#   ...
#   ✓ Sync completo y verificado.
#   🚀 Arrancando uvicorn...
```

Variables relevantes (todas tienen defaults, pero podés override en `.env` o docker-compose):

| Variable | Default | Comentario |
|---|---|---|
| `MODELS_DIR` | `/app/models` | Destino de los pesos extraídos |
| `MANIFEST_PATH` | `/app/release_manifest.json` | Manifest con URLs + SHAs |
| `DOWNLOAD_FAIL_POLICY` | `abort` | `abort` (sys.exit 1) o `warn` (loggea y arranca con fallback heurístico) |
| `MIN_FREE_DISK_GB` | `10` | Pre-check de espacio antes de bajar |

### Opción B — HuggingFace Hub (legacy, compat hacia atrás)

Útil si Drive no es accesible desde la red de tu instancia o si publicaste los modelos como repo privado en HF.

```bash
cp .env.example .env
# Editar .env:
#   DOWNLOAD_MODELS_ON_START=true
#   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#   HF_REPO_T1=tu-org/scibeto-es-t1
#   HF_REPO_T2=tu-org/scibeto-es-t2
#   HF_REPO_T2_CTX=tu-org/scibeto-es-t2-ctx   # (opcional)
```

Con `DOWNLOAD_MODELS_ON_START=true`, el `entrypoint.sh` corre `scripts/download_models.sh` en vez del `model_sync` de Drive. **No hay verificación SHA256** en este camino — usalo solo si confías 100% en HF.

### Opción C — S3 (legacy)

```bash
# 1. Subir los modelos a S3 (una vez):
aws s3 sync ./models/scibeto-es-t1 s3://maia-models/scibeto-es-t1/
aws s3 sync ./models/scibeto-es-t2   s3://maia-models/scibeto-es-t2/
aws s3 sync ./models/scibeto-es-t2-ctx s3://maia-models/scibeto-es-t2-ctx/

# 2. En la EC2, configurar .env:
#   DOWNLOAD_MODELS_ON_START=true
#   S3_BUCKET=maia-models
#   S3_PREFIX_T1=scibeto-es-t1/
#   S3_PREFIX_T2=scibeto-es-t2/
#   S3_PREFIX_T2_CTX=scibeto-es-t2-ctx/
```

La instancia EC2 necesita IAM role con `s3:GetObject` + `s3:ListBucket` sobre el bucket, o credenciales explícitas. Tampoco hay SHA check en este camino.

### Opción D — Copiar manualmente

Útil para iteración rápida en dev:

```bash
scp -i your-key.pem -r local-models/ ubuntu@<ec2-public-ip>:~/app/models/
# Layout esperado:
#   app/models/scibeto-es-t1/{config.json, model.safetensors, tokenizer*, ...}
#   app/models/scibeto-es-t2/{...}
#   app/models/scibeto-es-t2-ctx/{...}   # opcional
```

Aunque copies manualmente, el `model_sync ensure_all` corre igual al arranque y **verifica los SHAs**. Si los archivos copiados no coinciden con el manifest, los re-descarga (a menos que pases `DOWNLOAD_FAIL_POLICY=warn` o uses `start.sh --skip-sync`).

## 5. Levantar el servicio

```bash
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml logs -f maia
```

Verificar que la GPU se detectó:

```bash
docker compose -f docker-compose.gpu.yml exec maia \
    python -c "import torch; print('CUDA:', torch.cuda.is_available(), '· Device:', torch.cuda.get_device_name(0))"
# Esperado: CUDA: True · Device: Tesla T4
```

## 6. Smoke test post-deploy

`scripts/smoke_test.sh` corre 8 validaciones en orden:

| # | Check | Pasa cuando... |
|---|---|---|
| 1 | `/health` 200 | Backend levantó y FastAPI responde |
| 2 | Catálogo T2 lista los 7 IDs | `heuristic`, `scibeto-es-t2`, `scibeto-es-t2-ctx`, 2× Llama, 2× GPT |
| 3 | Catálogo T1 incluye `scibeto-es-t1` | Rename del Issue #1 aplicado correctamente |
| 4 | Pesos cargados (`available=true`) | No hay fallback silencioso a heurística |
| 5 | T1 inferencia real → `mode=encoder` | El encoder fine-tuned se está usando, no heurística |
| 6 | T2 inferencia real → `mode=encoder` | Encadenamiento T1→T2 funciona end-to-end |
| 7 | `model_sync verify` exit 0 | SHAs de archivos individuales coinciden con manifest |
| 8 | Latencia en caliente < 3 s | GPU + preload funcionando |

Uso:

```bash
# Desde la EC2 (después de docker compose up):
bash scripts/smoke_test.sh                          # contra localhost:8000
bash scripts/smoke_test.sh http://1.2.3.4:8000      # contra IP pública

# Desde tu laptop (vía túnel SSH o IP pública):
ssh -L 8000:localhost:8000 -i your-key.pem ubuntu@<ec2-ip> &
bash scripts/smoke_test.sh http://localhost:8000
```

**Output esperado en una EC2 sana:**

```
[1/8] Backend reachable y /health responde 200
  ✓ PASS — GET /health → 200
  ✓ PASS — status='ok' · version=2.1.0
[2/8] Catálogo T2 lista los 7 modelos esperados
  ✓ PASS — 7/7 IDs T2 presentes
[3/8] Catálogo T1 incluye scibeto-es-t1 (no 'scibert-es' legacy)
  ✓ PASS — scibeto-es-t1 presente en catálogo T1
[4/8] SciBETO T2 baseline available=true (pesos descargados)
  ✓ PASS — scibeto-es-t2 available=true
  ✓ PASS — scibeto-es-t2-ctx available=true
  ✓ PASS — scibeto-es-t1 available=true
[5/8] POST /api/segment con texto sample → mode=encoder
  ✓ PASS — POST /api/segment → 200 (350ms wall-clock)
  ✓ PASS — mode=encoder · model_name="SciBETO-large V5 fine-tuned" · n_segments=4
[6/8] POST /api/contributions usando segments de T1 → mode=encoder
  ✓ PASS — POST /api/contributions → 200 (180ms wall-clock)
  ✓ PASS — mode=encoder
  ✓ PASS — 2/4 fragments clasificados como contribución
[7/8] Verificación de integridad de pesos (SHA256 contra manifest)
  ✓ PASS — model_sync verify → exit 0 (todos los SHAs coinciden)
[8/8] Latencia request-en-caliente (segundo POST /api/segment)
  ✓ PASS — Request en caliente: 150ms (<3s, buena latencia)

  ✓ SMOKE TEST PASS — 0 fails
  Demo listo para uso público.
```

**Diagnóstico de fails típicos:**

| Fail | Causa probable | Acción |
|---|---|---|
| Check 1 — HTTP 000 | Servicio caído / puerto bloqueado | `docker compose ps` + revisar logs |
| Check 4 — `available=false` | Descarga de Drive falló silenciosa | `docker compose logs maia \| grep -i model_sync` |
| Check 5/6 — `mode=heuristic` | Encoder no cargó (pesos OK pero falla en runtime) | Revisar logs de `transformers` / VRAM |
| Check 7 — exit ≠ 0 | Corrupción silenciosa post-descarga | Borrar `models/.cache/` y `models/<id>/`, re-arrancar |
| Check 8 — > 10 s | Caída a CPU o `T*_PRELOAD=false` con cold start | Verificar `nvidia-smi` desde el contenedor |

## 7. Acceso público (seguridad)

Por defecto el servicio escucha en `0.0.0.0:8000`. Para exponerlo:

```bash
# Security Group: abrir 80/443 (no 8000 directamente al público)
sudo apt install -y nginx certbot python3-certbot-nginx
sudo tee /etc/nginx/sites-available/maia <<'EOF'
server {
    server_name demo.maia.tudominio.com;
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/maia /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d demo.maia.tudominio.com
```

## 8. Costos estimados

Para una demo activa unas pocas horas al día:

| Concepto | Costo/mes |
|---|---|
| g4dn.xlarge on-demand (4 h/día, 30 días) | ~$64 |
| g4dn.xlarge spot (mismo uso, ~70 % descuento) | ~$19 |
| Almacenamiento EBS (50 GB gp3) | ~$4 |
| Transferencia salida (1 GB/día) | ~$2.7 |
| **Total approx.** | **$25–70/mes** |

**Recomendación para la defensa:** levantar la instancia 1–2 h antes y bajarla después. `docker compose down` no apaga la VM — hay que `aws ec2 stop-instances --instance-ids i-xxxx`.

## 9. Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| `TypeError: download() got an unexpected keyword argument 'fuzzy'` | Versión vieja de `model_sync.py` con gdown >=6.0 instalado (gdown 6.0 eliminó el parámetro `fuzzy` — issue #455). | Actualizar `model_sync.py` a la versión incluida en este release (detecta la firma vía `inspect.signature` y omite `fuzzy` si gdown no lo soporta). El `requirements.txt` ahora pinea `gdown>=6.0.0,<7.0.0`. |
| `gdown: command not found` o `403 Forbidden` al bajar de Drive | Permisos del ZIP no son "Anyone with the link" | Verificar en Drive web: link → Compartir → "Cualquiera con el enlace" → "Lector" |
| `model_sync ensure_all` aborta con `Espacio insuficiente` | EBS < 10 GB libres | Aumentar el volume (`aws ec2 modify-volume`) o bajar `MIN_FREE_DISK_GB` |
| Hash mismatch persistente tras 3 reintentos | Manifest desactualizado vs ZIPs en Drive | Comparar SHA del ZIP local con `manifest.models[id].zip.sha256` |
| `RuntimeError: CUDA out of memory` | Múltiples encoders en VRAM | Bajar `T1_BATCH_SIZE` y `T2_BATCH_SIZE` (default 32 → 16 → 8) |
| Primer request > 30 s | Modelo cargándose | Setear `T1_PRELOAD=true` y `T2_PRELOAD=true` para precargar |
| Frontend dice "fallback heurístico" | Pesos OK en disco pero encoder no cargó | Revisar logs por `ModelNotAvailableError` |
| `model_sync verify` falla en cada arranque | Filesystem corrupto o pesos modificados manualmente | Borrar `models/<id>/` y re-arrancar; descargará de cero |

## 10. Configuración mínima viable

El servicio funciona sin TODOS los modelos descargados — el catálogo marca `available=false` y los endpoints hacen fallback transparente:

- Solo `scibeto-es-t1` (T1) → segmentación con encoder, contribuciones con heurística.
- Solo `scibeto-es-t2` (T2) → segmentación con heurística, contribuciones con encoder.
- Sin modelos → demo 100 % heurístico (siempre funciona, F1 más bajo).

Para deshabilitar la descarga completamente (modo full-heuristic), correr con `DOWNLOAD_FAIL_POLICY=warn` y `MODELS_DIR=/tmp/empty`.

## 11. Re-arranques rápidos

```bash
# Re-arranque sin bajar nada (verifica SHAs y arranca, ~5-10 s):
docker compose -f docker-compose.gpu.yml restart maia

# Re-arranque saltando incluso la verificación (no recomendado en prod):
docker compose -f docker-compose.gpu.yml exec maia \
    bash -c "cd /app && ./start.sh --skip-sync --skip-install --no-reload"

# Forzar re-descarga de un modelo específico (después de borrar el dir):
docker compose -f docker-compose.gpu.yml exec maia \
    rm -rf /app/models/scibeto-es-t2
docker compose -f docker-compose.gpu.yml restart maia
```
