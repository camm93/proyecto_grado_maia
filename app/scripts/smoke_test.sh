#!/usr/bin/env bash
# smoke_test.sh -- Validacion end-to-end del demo MAIA post-deploy.
# ===============================================================================
# Corre 8 checks contra una instancia recien levantada. Cada check imprime
# PASS/FAIL con razon. Exit code = cantidad de fails (0 = todo OK).
#
# Uso:
#   bash scripts/smoke_test.sh                    # contra localhost:8000
#   bash scripts/smoke_test.sh http://1.2.3.4:8000
#   bash scripts/smoke_test.sh https://demo.maia.tudominio.com
#
# Requiere: curl, jq (apt install -y jq).
# Tiempo total esperado: 20-40 s (el T1+T2 paga la carga de modelos si
# T*_PRELOAD=false).
# ===============================================================================
set -uo pipefail   # NO `-e`: queremos seguir hasta el final aunque un check falle

BASE_URL="${1:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"   # quita slash final si lo hay
FAILS=0

# -- Colores (deshabilitados si no es TTY) ------------------------------------
if [[ -t 1 ]]; then
    R='\033[0;31m'; G='\033[0;32m'; Y='\033[0;33m'; B='\033[0;34m'; N='\033[0m'
else
    R=''; G=''; Y=''; B=''; N=''
fi

# -- Pre-flight: curl + jq ----------------------------------------------------
command -v curl >/dev/null || { echo "ERROR: curl no instalado"; exit 99; }
command -v jq   >/dev/null || { echo "ERROR: jq no instalado (apt install -y jq)"; exit 99; }

echo -e "${B}===============================================================${N}"
echo -e "${B}  MAIA Smoke Test -- $BASE_URL${N}"
echo -e "${B}  $(date -u '+%Y-%m-%d %H:%M:%S UTC')${N}"
echo -e "${B}===============================================================${N}"

# -- Helpers ----------------------------------------------------------------
pass() { echo -e "  ${G}[ OK ]${N} $1"; }
fail() { echo -e "  ${R}[FAIL]${N} $1"; FAILS=$((FAILS + 1)); }
warn() { echo -e "  ${Y}[WARN]${N} $1"; }
step() { echo; echo -e "${B}[$1]${N} $2"; }

# Tmp para responses
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# ============================================================================
# Check 1 -- Reachability
# ============================================================================
step 1/8 "Backend reachable y /health responde 200"

http_code=$(curl -s -o "$TMP/health.json" -w "%{http_code}" \
            --max-time 10 "$BASE_URL/health")
http_code="${http_code:-000}"

if [[ "$http_code" == "200" ]]; then
    pass "GET /health -> 200"
    status=$(jq -r '.status // "missing"' "$TMP/health.json")
    version=$(jq -r '.version // "missing"' "$TMP/health.json")
    if [[ "$status" == "ok" ]]; then
        pass "status='ok' | version=$version"
    else
        fail "status='$status' (esperado 'ok')"
    fi
else
    fail "GET /health -> HTTP $http_code (servicio caido? URL incorrecta?)"
    echo -e "${R}  Abortando -- el resto de los checks requieren backend vivo.${N}"
    exit $FAILS
fi

# ============================================================================
# Check 2 -- Catalogo T2 completo (7 IDs)
# ============================================================================
step 2/8 "Catalogo T2 lista los 7 modelos esperados"

curl -s --max-time 10 "$BASE_URL/api/models?task=T2" > "$TMP/t2.json"
got_ids=$(jq -r '.models[].id' "$TMP/t2.json" | sort | paste -sd',' -)
expected_ids="gpt-4o-mini-t2,gpt-4o-mini-t2-fs8,heuristic,llama-3.1-8b-t2,llama-3.1-8b-t2-fs8,scibeto-es-t2,scibeto-es-t2-ctx"

if [[ "$got_ids" == "$expected_ids" ]]; then
    pass "7/7 IDs T2 presentes"
else
    fail "IDs T2 no coinciden"
    echo "    esperado: $expected_ids"
    echo "    obtenido: $got_ids"
fi

# ============================================================================
# Check 3 -- Catalogo T1 incluye scibeto-es-t1 (rename Issue #1)
# ============================================================================
step 3/8 "Catalogo T1 incluye scibeto-es-t1 (no 'scibert-es' legacy)"

curl -s --max-time 10 "$BASE_URL/api/models?task=T1" > "$TMP/t1.json"
if jq -e '.models[] | select(.id == "scibeto-es-t1")' "$TMP/t1.json" >/dev/null; then
    pass "scibeto-es-t1 presente en catalogo T1"
else
    fail "scibeto-es-t1 NO encontrado (rename del Issue #1 no se aplico?)"
fi

if jq -e '.models[] | select(.id == "scibert-es")' "$TMP/t1.json" >/dev/null; then
    fail "ID legacy 'scibert-es' todavia presente -- deberias haber renombrado"
fi

# ============================================================================
# Check 4 -- Pesos cargados (no fallback silencioso)
# ============================================================================
step 4/8 "SciBETO T2 baseline available=true (pesos descargados)"

t2_avail=$(jq -r '.models[] | select(.id == "scibeto-es-t2") | .available' "$TMP/t2.json")
t2ctx_avail=$(jq -r '.models[] | select(.id == "scibeto-es-t2-ctx") | .available' "$TMP/t2.json")
t1_avail=$(jq -r '.models[] | select(.id == "scibeto-es-t1") | .available' "$TMP/t1.json")

if [[ "$t2_avail" == "true" ]]; then
    pass "scibeto-es-t2 available=true"
else
    fail "scibeto-es-t2 available=false -- pesos no se cargaron o fallo model_sync"
fi

if [[ "$t2ctx_avail" == "true" ]]; then
    pass "scibeto-es-t2-ctx available=true"
else
    warn "scibeto-es-t2-ctx available=false (ablation; opcional para el demo)"
fi

if [[ "$t1_avail" == "true" ]]; then
    pass "scibeto-es-t1 available=true"
else
    fail "scibeto-es-t1 available=false -- T1 no usara encoder fine-tuned"
fi

# ============================================================================
# Check 5 -- Inferencia real T1 (mode=encoder, no heuristic)
# ============================================================================
step 5/8 "POST /api/segment con texto sample -> mode=encoder"

SAMPLE_TEXT='En este trabajo proponemos un nuevo metodo de segmentacion retorica '\
'basado en SciBETO-large fine-tuned sobre un corpus de articulos cientificos '\
'en espanol. Nuestro enfoque obtiene un F1 macro de 0.40 sobre el conjunto '\
'de prueba gold humano, superando a la linea base heuristica en 8 puntos '\
'porcentuales. Como trabajo futuro proponemos extender la evaluacion a otros '\
'dominios academicos como las ciencias sociales y las humanidades.'

t1_req=$(jq -n --arg t "$SAMPLE_TEXT" \
            '{text: $t, model_id: "scibeto-es-t1"}')

t1_t0=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
http_code=$(curl -s -o "$TMP/segment.json" -w "%{http_code}" \
            --max-time 60 -X POST "$BASE_URL/api/segment" \
            -H 'Content-Type: application/json' -d "$t1_req")
t1_t1=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
t1_elapsed=$((t1_t1 - t1_t0))

if [[ "$http_code" != "200" ]]; then
    fail "POST /api/segment -> HTTP $http_code"
    jq '.' "$TMP/segment.json" 2>/dev/null | head -10
else
    pass "POST /api/segment -> 200 (${t1_elapsed}ms wall-clock)"
    t1_mode=$(jq -r '.mode' "$TMP/segment.json")
    t1_name=$(jq -r '.model_name' "$TMP/segment.json")
    n_seg=$(jq -r '.segments | length' "$TMP/segment.json")
    if [[ "$t1_mode" == "encoder" ]]; then
        pass "mode=encoder | model_name=\"$t1_name\" | n_segments=$n_seg"
    else
        fail "mode='$t1_mode' (esperado 'encoder'). Backend cayo a fallback: \"$t1_name\""
    fi
fi

# ============================================================================
# Check 6 -- Inferencia T2 con segments de T1 (encadenamiento real)
# ============================================================================
step 6/8 "POST /api/contributions usando segments de T1 -> mode=encoder"

if [[ -s "$TMP/segment.json" ]] && [[ "$http_code" == "200" ]]; then
    t2_req=$(jq -n --argjson frags "$(jq '.segments' "$TMP/segment.json")" \
                '{fragments: $frags, model_id: "scibeto-es-t2"}')

    t2_t0=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
    http_code=$(curl -s -o "$TMP/contrib.json" -w "%{http_code}" \
                --max-time 60 -X POST "$BASE_URL/api/contributions" \
                -H 'Content-Type: application/json' -d "$t2_req")
    t2_t1=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
    t2_elapsed=$((t2_t1 - t2_t0))

    if [[ "$http_code" == "200" ]]; then
        pass "POST /api/contributions -> 200 (${t2_elapsed}ms wall-clock)"
        t2_mode=$(jq -r '.mode' "$TMP/contrib.json")
        t2_name=$(jq -r '.model_name' "$TMP/contrib.json")
        n_detected=$(jq -r '.summary.contributions_detected' "$TMP/contrib.json")
        n_total=$(jq -r '.summary.total_fragments' "$TMP/contrib.json")
        if [[ "$t2_mode" == "encoder" ]]; then
            pass "mode=encoder | model_name=\"$t2_name\""
            pass "$n_detected/$n_total fragments clasificados como contribucion"
        else
            fail "mode='$t2_mode' (esperado 'encoder'). Fallback: \"$t2_name\""
        fi
    else
        fail "POST /api/contributions -> HTTP $http_code"
    fi
else
    fail "Skipping T2 -- el check 5 fallo y no hay segments validos"
fi

# ============================================================================
# Check 7 -- Integridad de pesos (model_sync verify dentro del contenedor)
# ============================================================================
step 7/8 "Verificacion de integridad de pesos (SHA256 contra manifest)"

# Si estamos en una EC2 con docker compose, podemos hacer exec; si estamos
# remotos, este check requiere SSH. Lo intentamos via docker si esta local.
if command -v docker >/dev/null 2>&1 && \
   docker compose ps 2>/dev/null | grep -q maia; then
    if docker compose exec -T maia python -m backend.utils.model_sync verify \
            > "$TMP/verify.log" 2>&1; then
        pass "model_sync verify -> exit 0 (todos los SHAs coinciden)"
    else
        fail "model_sync verify -> exit $? (hash mismatch o archivo faltante)"
        tail -5 "$TMP/verify.log" | sed 's/^/    /'
    fi
else
    warn "Saltado -- sin acceso a docker compose desde aca."
    warn "Correr manualmente en la EC2: docker compose exec maia python -m backend.utils.model_sync verify"
fi

# ============================================================================
# Check 8 -- Sanity de latencia (request en caliente)
# ============================================================================
step 8/8 "Latencia request-en-caliente (segundo POST /api/segment)"

t_warm_0=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 30 -X POST "$BASE_URL/api/segment" \
            -H 'Content-Type: application/json' -d "$t1_req")
t_warm_1=$(date +%s%3N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1000))')
warm_elapsed=$((t_warm_1 - t_warm_0))

if [[ "$http_code" == "200" ]]; then
    if [[ $warm_elapsed -lt 3000 ]]; then
        pass "Request en caliente: ${warm_elapsed}ms (<3s, buena latencia)"
    elif [[ $warm_elapsed -lt 10000 ]]; then
        warn "Request en caliente: ${warm_elapsed}ms (lento; T*_PRELOAD=false? CPU en vez de GPU?)"
    else
        fail "Request en caliente: ${warm_elapsed}ms (muy lento, revisar GPU/preload)"
    fi
else
    fail "Segundo /api/segment -> HTTP $http_code"
fi

# ============================================================================
# Resumen
# ============================================================================
echo
echo -e "${B}===============================================================${N}"
if [[ $FAILS -eq 0 ]]; then
    echo -e "${G}  [OK] SMOKE TEST PASS -- 0 fails${N}"
    echo -e "${G}  Demo listo para uso publico.${N}"
else
    echo -e "${R}  [X] SMOKE TEST FAIL -- $FAILS check(s) fallaron${N}"
    echo -e "${Y}  Revisar logs del contenedor: docker compose logs maia${N}"
fi
echo -e "${B}===============================================================${N}"

exit $FAILS
