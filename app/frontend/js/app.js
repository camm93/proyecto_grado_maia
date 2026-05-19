const API=(location.hostname==='localhost'||location.hostname==='127.0.0.1')
  ?'http://127.0.0.1:8000':'';

// ── PDF.js worker — configurar lo antes posible ──────────────────────────────
// Cuando se abre el HTML desde file:// los browsers bloquean la carga de workers
// externos por política de seguridad de origen único. En ese caso se deshabilita
// el worker (modo síncrono, más lento pero funcional para documentos pequeños).
// Con un servidor HTTP (localhost o producción) se usa el worker CDN normalmente.
(function initPdfWorker(){
  const CDN='https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  if(location.protocol==='file:'){
    // Fallback seguro para file://: worker vacío (PDF.js lo maneja internamente)
    if(typeof pdfjsLib!=='undefined') pdfjsLib.GlobalWorkerOptions.workerSrc='';
    // Reintentar cuando pdfjsLib cargue si aún no está disponible
    window.addEventListener('load',()=>{
      if(typeof pdfjsLib!=='undefined') pdfjsLib.GlobalWorkerOptions.workerSrc='';
    });
  } else {
    if(typeof pdfjsLib!=='undefined') pdfjsLib.GlobalWorkerOptions.workerSrc=CDN;
    window.addEventListener('load',()=>{
      if(typeof pdfjsLib!=='undefined') pdfjsLib.GlobalWorkerOptions.workerSrc=CDN;
    });
  }
})();
/* ── Tema ── */
function toggleTheme(){
  const h=document.documentElement, d=h.dataset.theme==='dark';
  h.dataset.theme=d?'light':'dark';
  document.getElementById('tbtn').textContent=d?'☀️':'🌙';
  localStorage.setItem('maia_t',h.dataset.theme);
}
(function(){
  // v7: el default del HTML es light. Si el usuario tiene preferencia
  // guardada en localStorage la respetamos; si no, dejamos el data-theme
  // que ya viene del HTML. En ambos casos sincronizamos el emoji al final
  // para que NUNCA quede desfasado (bug: si solo lee de localStorage y no
  // hay nada guardado, el emoji puede quedar inconsistente con el theme).
  const s=localStorage.getItem('maia_t');
  if(s) document.documentElement.dataset.theme=s;
  const current=document.documentElement.dataset.theme;
  document.getElementById('tbtn').textContent=current==='light'?'☀️':'🌙';
})();
/* ── Aviso vista no optimizada (v7.6) ──────────────────────────────────────
   Logica de visibilidad del aviso soft para mobile/tablet (<900px). La
   deteccion de tamano la hace 100% el CSS via @media; aqui solo manejamos
   la persistencia de la decision del usuario en localStorage y la clase
   .mn-dismissed que controla el toggle overlay <-> badge.
   La clave incluye el sufijo de version (v76) para que, si en una version
   futura optimizamos la UI mobile, podamos invalidar todos los opt-outs
   simplemente cambiando el sufijo. */
function _dismissMobileNotice(){
  document.body.classList.add('mn-dismissed');
  try{localStorage.setItem('maia_mn_dismissed_v76','1');}catch(e){}
}
function _reshowMobileNotice(){
  document.body.classList.remove('mn-dismissed');
  try{localStorage.removeItem('maia_mn_dismissed_v76');}catch(e){}
}
(function(){
  // Aplicar la clase ANTES del primer paint si el usuario ya cerro el
  // aviso en una visita anterior. Si el viewport es >=900px la @media de
  // CSS oculta el overlay igual sin importar esta clase, asi que es
  // seguro setearla siempre.
  try{
    if(localStorage.getItem('maia_mn_dismissed_v76')==='1'){
      document.body.classList.add('mn-dismissed');
    }
  }catch(e){}
})();
/* ── Utility: detectar si un model_id es un LLM (v7.7) ────────────────────
   Los LLMs (Llama y GPT) corren en modo greedy decoding y NO devuelven
   probabilidades calibradas. Su "confidence" en el backend es siempre 1.0
   (o 0.5 si hubo error de parseo), no una metrica de incertidumbre real.
   Mostrar ese 100% en la UI es enganyoso, asi que las funciones de render
   usan este predicado para suprimir la barra+% en los tooltips cuando el
   modelo activo es LLM. Heuristico y SciBETO si exponen probs (pseudo y
   reales respectivamente) — se siguen mostrando como antes. */
function _isLLM(modelId){
  return typeof modelId==='string' && /^(llama|gpt)-/.test(modelId);
}
/* ── Meta retórica ── 
   Los colores y nombres se cargan dinámicamente desde /api/lexicon en compileLexicon().
   El acceso es vía LEX.t1Meta[label].{rgb, n}. El fallback en LEX_FALLBACK garantiza
   que el demo funcione aunque el backend esté caído. */
let selModel='heuristic';   // T1 — segmentación retórica
let selModel2='heuristic';  // T2 — detección de contribuciones
let _pageFormat='Carta';
// La leyenda se renderiza dentro del modal de ayuda (ver openHelp())

/* ── Health ── */
async function chkH(){
  const p=document.getElementById('spill'),t=document.getElementById('stxt');
  try{await fetch(API+'/health',{signal:AbortSignal.timeout(3000)});
    p.className='spill';t.textContent='backend activo';}
  catch{p.className='spill off';t.textContent='sin backend — modo local';}
}

/* ── Modelos ── */
// Catálogo compartido — T1 y T2 pueden usar modelos distintos.
// Familias: 'heuristic' (baseline reglas), 'encoder' (BERT-like fine-tuned),
// 'llm_open' (Llama/Mistral), 'llm_api' (GPT-4o/Gemini). El backend es la
// fuente de verdad: este array solo se usa si /api/models no responde.
const MODEL_CATALOG_FALLBACK=[
  {id:'heuristic',  available:true, name:'Heurístico (sin modelo)',family:'heuristic',
   description:'Reglas léxicas + posición relativa. Siempre disponible.'},
  {id:'scibeto-es-t1', available:false,name:'SciBETO-large',          family:'encoder',
   description:'BETO fine-tuned en corpus científico español. Pendiente A5.'},
  {id:'mdeberta-v3',available:false,name:'mDeBERTa-v3-base',       family:'encoder',
   description:'DeBERTa multilingüe v3, cross-lingual.'},
  {id:'llama-3.1-8b',available:false,name:'Llama 3.1 8B',          family:'llm_open',
   description:'Meta Llama 3.1 8B zero-shot. Requiere GPU ≥16 GB.'},
  {id:'mistral-7b', available:false,name:'Mistral 7B',             family:'llm_open',
   description:'Mistral 7B Instruct, cuantizado CPU.'},
  {id:'gpt-4o',     available:false,name:'GPT-4o (API)',            family:'llm_api',
   description:'OpenAI GPT-4o. Requiere OPENAI_API_KEY.'},
  {id:'gemini-pro', available:false,name:'Gemini 1.5 Pro (API)',    family:'llm_api',
   description:'Google Gemini 1.5 Pro. Requiere GOOGLE_API_KEY.'},
];

function buildSelectOptions(sel, ms, currentId){
  sel.innerHTML='';
  // v7: el heuristico ya NO aparece en el dropdown; sigue activo como
  // fallback silencioso si el modelo elegido no esta disponible (el banner
  // superior reporta el modo real usado al terminar el analisis).
  // Esto libera espacio para los otros modelos del catalogo.
  const visibleFamilies=[
    {key:'encoder', label:'Encoders fine-tuned'},
    {key:'llm_open',label:'LLM open-weight'},
    {key:'llm_api', label:'LLM via API'},
  ];
  visibleFamilies.forEach(f=>{
    const fm=ms.filter(m=>m.family===f.key);if(!fm.length)return;
    const g=document.createElement('optgroup');g.label=f.label;
    fm.forEach(m=>{
      const o=document.createElement('option');o.value=m.id;
      o.textContent=(m.available?'● ':'○ ')+m.name;
      if(m.id===currentId)o.selected=true;
      g.appendChild(o);
    });sel.appendChild(g);
  });
}

// v7: orden de preferencia para auto-seleccionar el "mejor modelo" en cada tarea.
// Refleja las metricas finales del notebook v3 (T1) y v5 (T2): el primer
// modelo disponible de la lista se selecciona por defecto. Si ninguno esta
// disponible cae a heuristico (que NO aparece en el dropdown pero se usa
// silenciosamente como fallback al analizar).
// v7.8.1: SciBETO movido al top por pedido de los evaluadores. Aunque
// GPT-4o-mini-t1-zs tiene mejor F1 (0.447 vs 0.400), priorizar el encoder
// local fine-tuned tiene sentido para una demo academica: es el modelo
// "propio" del proyecto de grado, es 100% local (sin API key ni costos
// por inferencia), y la diferencia de F1 (4.7 pp) no compensa la
// dependencia externa para el caso de uso de defensa/demo.
const T1_PREFERENCE = [
  'scibeto-es-t1',       // F1 0.400 - encoder local fine-tuned (DEFAULT)
  'gpt-4o-mini-t1-zs',   // F1 0.447 - mejor F1 absoluto pero requiere API
  'gpt-4o-mini-t1',      // F1 0.414 - FS-14
  'llama-3.1-8b-t1',     // F1 0.322 - FS-14
  'llama-3.1-8b-t1-zs',  // F1 0.234 - ZS (ablacion)
];
const T2_PREFERENCE = [
  'scibeto-es-t2',       // F1 0.852 - mejor T2 por amplio margen (text-only)
  'scibeto-es-t2-ctx',   // F1 0.798 - ablacion con prefijo retorico
  'gpt-4o-mini-t2',      // F1 0.423 - ZS
  'gpt-4o-mini-t2-fs8',  // F1 0.380 - FS k=8
  'llama-3.1-8b-t2',     // F1 0.393 - ZS
  'llama-3.1-8b-t2-fs8', // F1 0.368 - FS k=8
];

function pickBestAvailable(catalog, preferenceList){
  // Recorre la lista de preferencia y devuelve el primer ID disponible.
  // Si ninguno esta disponible, devuelve el TOP de la lista de preferencia
  // (sin importar disponibilidad) para que el dropdown muestre algo
  // coherente. El backend hara fallback silencioso a heuristico al analizar
  // y el banner superior reportara el modo real ('heuristic' o '[fallback
  // heuristico]') tras terminar el analisis (ver catalog.py::resolve_model).
  for(const id of preferenceList){
    const m=catalog.find(x=>x.id===id);
    if(m && m.available) return id;
  }
  return preferenceList[0];
}

async function loadModels(){
  let msT1, msT2;
  try{
    const r=await fetch(API+'/api/models');
    const d=await r.json();
    // Nuevo backend devuelve {T1:[...], T2:[...]}
    msT1=d.T1||d.models||MODEL_CATALOG_FALLBACK;
    msT2=d.T2||d.models||MODEL_CATALOG_FALLBACK;
  }catch{
    msT1=msT2=MODEL_CATALOG_FALLBACK;
  }
  window._msT1=msT1; window._msT2=msT2;
  // v7: auto-elegir el mejor disponible para cada tarea (gpt-4o-mini-t1-zs
  // si esta arriba, scibeto-es-t2 para T2). Si ninguno del catalogo esta
  // disponible cae a heuristico silenciosamente.
  selModel=pickBestAvailable(msT1, T1_PREFERENCE);
  selModel2=pickBestAvailable(msT2, T2_PREFERENCE);
  buildSelectOptions(document.getElementById('msel'), msT1, selModel);
  onMC(document.getElementById('msel'));
  buildSelectOptions(document.getElementById('msel2'), msT2, selModel2);
  onMC2(document.getElementById('msel2'));
}

function _updateModelUI(modelId, descId, availId, catalog){
  const m=(catalog||[]).find(x=>x.id===modelId);if(!m)return;
  document.getElementById(descId).textContent=m.description;
  document.getElementById(availId).innerHTML=m.available
    ?'<span class="aok">●</span><span class="aok">Disponible</span>'
    :'<span class="apend">○</span><span class="apend">Pendiente — fallback heurístico</span>';
}

function onMC(sel){
  selModel=sel.value;
  _updateModelUI(selModel,'mdesc','mavail',window._msT1);
}

function onMC2(sel){
  selModel2=sel.value;
  _updateModelUI(selModel2,'mdesc2','mavail2',window._msT2);
}
/* ── Detección de formato de página ── */
// PDF.js devuelve dimensiones en puntos (1pt = 1/72 inch)
// Tolerancia de ±5pts para absorber variaciones de exportadores
function detectFormatFromPts(wPts, hPts){
  // Normalizar a portrait (ancho ≤ alto)
  const w=Math.min(wPts,hPts), h=Math.max(wPts,hPts);
  if(Math.abs(w-595)<=8 && Math.abs(h-842)<=8) return 'A4';
  if(Math.abs(w-612)<=8 && Math.abs(h-1008)<=8) return 'Oficio';
  if(Math.abs(w-612)<=8 && Math.abs(h-936)<=8)  return 'Folio';
  if(Math.abs(w-612)<=8 && Math.abs(h-792)<=8)  return 'Carta';
  // Fallback: si alto > 1000pts es Oficio, si 850-1000 es Folio
  if(h>1000) return 'Oficio';
  if(h>870)  return 'Folio';
  if(h>820)  return 'A4';
  return 'Carta';
}

// Intenta leer el tamaño de página de un DOCX (w:pgSz en word/document.xml)
// DOCX es un ZIP; leemos los primeros bytes buscando la cadena w:pgSz
async function detectFormatFromDocx(buf){
  try{
    // Buscar "w:pgSz" en el buffer como texto (está en word/document.xml dentro del ZIP)
    const bytes=new Uint8Array(buf);
    const str=new TextDecoder('utf-8',{fatal:false}).decode(bytes);
    const m=str.match(/w:pgSz\s+[^>]*w:w="(\d+)"[^>]*w:h="(\d+)"/);
    if(!m){
      // intentar orden inverso (h antes que w)
      const m2=str.match(/w:pgSz\s+[^>]*w:h="(\d+)"[^>]*w:w="(\d+)"/);
      if(m2){
        const hTwips=parseInt(m2[1]), wTwips=parseInt(m2[2]);
        return detectFormatFromPts(wTwips/20, hTwips/20);
      }
      return 'Carta';
    }
    const wTwips=parseInt(m[1]), hTwips=parseInt(m[2]);
    // 1 twip = 1/20 pt
    return detectFormatFromPts(wTwips/20, hTwips/20);
  }catch{ return 'Carta'; }
}

/* ══════════════════════════════════════════════════════════════════════════
   EXTRACTOR PDF TOLERANTE A COLUMNAS
   ══════════════════════════════════════════════════════════════════════════
   Estrategia:
   1. Obtener todos los ítems de texto con sus coordenadas (x, y, height)
   2. Filtrar ruido: encabezados/pies de página fuera del área de contenido
   3. Detectar número de columnas mediante clustering 1D sobre coordenadas X
      usando k-means simplificado con k=1,2,3 y eligiendo el que minimiza
      la varianza intra-cluster (criterio del codo)
   4. Asignar cada ítem a su columna
   5. Dentro de cada columna, ordenar por Y descendente (arriba → abajo)
   6. Reconstruir el texto columna por columna, insertando saltos de párrafo
      cuando la separación vertical entre ítems supera 1.5× la altura media
      de línea (gap de párrafo)
   ══════════════════════════════════════════════════════════════════════════ */

/* ── Utilidades de clustering 1D ── */
function kmeans1D(values, k, maxIter=20){
  if(values.length===0) return {labels:[], centroids:[]};
  // Inicializar centroides uniformemente en el rango
  const mn=Math.min(...values), mx=Math.max(...values);
  let centroids=Array.from({length:k},(_,i)=>mn+(mx-mn)*i/(Math.max(k-1,1)));
  let labels=new Array(values.length).fill(0);

  for(let iter=0;iter<maxIter;iter++){
    // Asignar cada punto al centroide más cercano
    const newLabels=values.map(v=>{
      let best=0, bestD=Infinity;
      centroids.forEach((c,ci)=>{const d=Math.abs(v-c);if(d<bestD){bestD=d;best=ci;}});
      return best;
    });
    // Actualizar centroides
    const newCentroids=centroids.map((_,ci)=>{
      const pts=values.filter((_,i)=>newLabels[i]===ci);
      return pts.length?pts.reduce((a,b)=>a+b,0)/pts.length:centroids[ci];
    });
    // Ordenar centroides (y reetiquetas) de izquierda a derecha
    const order=[...newCentroids.keys()].sort((a,b)=>newCentroids[a]-newCentroids[b]);
    const remapped=Object.fromEntries(order.map((orig,ni)=>[orig,ni]));
    const sortedLabels=newLabels.map(l=>remapped[l]);
    const sortedCentroids=order.map(i=>newCentroids[i]);

    const changed=sortedLabels.some((l,i)=>l!==labels[i]);
    labels=sortedLabels; centroids=sortedCentroids;
    if(!changed) break;
  }
  return {labels, centroids};
}

function inertia1D(values, labels, centroids){
  return values.reduce((sum,v,i)=>sum+(v-centroids[labels[i]])**2, 0);
}

/* ── Detección automática de número de columnas ── */
function detectColumns(items, pageWidth){
  if(items.length<4) return 1;

  // Usar la coordenada X del inicio de cada ítem
  const xs=items.map(it=>it.transform[4]);

  // Filtrar ítems muy cortos (ruido) para la decisión de columnas
  const meaningful=items.filter(it=>it.str.trim().length>3);
  if(meaningful.length<4) return 1;
  const mxs=meaningful.map(it=>it.transform[4]);

  // Probar k=1,2,3 y elegir por criterio del codo
  const results=[];
  for(let k=1;k<=3;k++){
    const {labels,centroids}=kmeans1D(mxs,k);
    const iner=inertia1D(mxs,labels,centroids);
    // Penalizar columnas muy estrechas (centroide muy cercano al anterior)
    const centDists=centroids.slice(1).map((c,i)=>c-centroids[i]);
    const minCentDist=Math.min(...centDists, pageWidth);
    const valid=k===1||(minCentDist>pageWidth*0.15); // columna mínimo 15% del ancho
    results.push({k,iner,centroids,valid});
  }

  // Elegir el mayor k válido donde la reducción de inercia es significativa (>30%)
  let best=1;
  for(let i=1;i<results.length;i++){
    if(!results[i].valid) continue;
    const reduction=(results[i-1].iner-results[i].iner)/Math.max(results[i-1].iner,1);
    if(reduction>0.30) best=results[i].k;
  }
  return best;
}

/* ── Reconstrucción de texto con soporte multicolumna ── */
function extractPageText(items, pageWidth, pageHeight){
  if(!items||items.length===0) return '';

  // 1. Calcular altura media de línea para umbral de párrafo
  const heights=items.map(it=>it.height||10).filter(h=>h>2);
  const avgH=heights.length?heights.reduce((a,b)=>a+b,0)/heights.length:12;
  const paraGap=avgH*1.8; // 1.8× interlineado = brecha real entre párrafos

  // 2. Excluir encabezados y pies: ítems en el 7% superior o inferior de la página
  const marginV=pageHeight*0.07;
  const body=items.filter(it=>{
    const y=it.transform[5];
    return y>marginV && y<pageHeight-marginV && it.str.trim().length>0;
  });
  if(!body.length) return items.map(it=>it.str).join(' ');

  // 3. Detectar número de columnas
  const nCols=detectColumns(body, pageWidth);

  if(nCols===1){
    // Una sola columna: ordenar por Y desc (arriba primero en PDF coords)
    const sorted=[...body].sort((a,b)=>b.transform[5]-a.transform[5]);
    return buildText(sorted, paraGap);
  }

  // 4. Multicolumna: asignar cada ítem a su columna via k-means
  const xs=body.map(it=>it.transform[4]);
  const {labels,centroids}=kmeans1D(xs, nCols);

  // Agrupar por columna (ya ordenadas izq→der por centroides)
  const cols=Array.from({length:nCols},()=>[]);
  body.forEach((it,i)=>cols[labels[i]].push(it));

  // 5. Dentro de cada columna, ordenar arriba → abajo
  cols.forEach(col=>col.sort((a,b)=>b.transform[5]-a.transform[5]));

  // 6. Concatenar columnas de izquierda a derecha
  return cols.map(col=>buildText(col,paraGap)).filter(t=>t.trim()).join('\n\n');
}

/* Construye texto desde ítems ya ordenados, insertando \n\n solo en gaps reales de párrafo.
   En PDFs a 2 columnas el interlineado normal es ~1× avgH.
   Un salto de párrafo tiene gap ≥ 1.8× avgH (espacio extra entre párrafos).
*/
function buildText(items, paraGap){
  if(!items.length) return '';
  // Recalcular avgH localmente para ser más preciso dentro de la columna
  const hs=items.map(it=>it.height||10).filter(h=>h>2);
  const avgH=hs.length?hs.reduce((a,b)=>a+b,0)/hs.length:10;
  // Umbral de párrafo: 1.8× interlineado. Nunca menos de 4pt.
  const paraThr=Math.max(4, avgH*1.8);
  let text='', prevY=null;
  for(const it of items){
    const y=it.transform[5];
    if(prevY!==null){
      const dy=prevY-y; // positivo = bajamos en la página
      if(dy>paraThr){
        text+='\n\n';          // gap de párrafo → nuevo párrafo
      } else if(dy>avgH*0.4){
        // Salto de línea normal dentro del mismo párrafo → espacio
        // Fusionar palabras separadas por guión tipográfico
        const lastChar=text.trimEnd().slice(-1);
        if(lastChar==='-') text=text.trimEnd().slice(0,-1);
        else if(!text.endsWith(' ')) text+=' ';
      } else {
        // Mismo renglón o solapamiento vertical (subíndices, superíndices)
        if(!text.endsWith(' ')&&it.str.length>0) text+=' ';
      }
    }
    text+=it.str;
    prevY=y;
  }
  return text.trim();
}

/* ── Archivos ── */
async function handleFile(e){
  const file=e.target.files[0];if(!file)return;
  // Si hay análisis activo, pedir confirmación antes de procesar el nuevo archivo
  if(_needsConfirm()){
    _showClearModal(
      `Se borrará el análisis actual para cargar "${file.name}". ¿Continuar?`,
      () => _processFile(file)
    );
    // Resetear el input para que el mismo archivo pueda volver a seleccionarse
    e.target.value='';
    return;
  }
  _processFile(file);
}

async function _processFile(file){
  const ext=file.name.split('.').pop().toLowerCase();
  const zone=document.getElementById('upz');
  zone.querySelector('.uptxt').innerHTML='<strong style="color:var(--acc)">'+file.name+'</strong><br><span style="font-size:.68rem">Leyendo…</span>';
  let text='';
  _pageFormat='Carta';
  try{
    if(ext==='txt'){
      text=await file.text();
      _pageFormat='Carta';
    }
    else if(ext==='pdf'){
      const buf=await file.arrayBuffer();
      const pdf=await pdfjsLib.getDocument({data:buf}).promise;
      // Formato desde la primera página
      const pg1=await pdf.getPage(1);
      const vp=pg1.view; // [x0,y0,width,height] en puntos
      _pageFormat=detectFormatFromPts(vp[2],vp[3]);
      const pageW=vp[2], pageH=vp[3];

      // Número de columnas detectado (mostrar en badge)
      let detectedCols=1;
      const pages=[];
      for(let i=1;i<=pdf.numPages;i++){
        const pg=await pdf.getPage(i);
        const tc=await pg.getTextContent({normalizeWhitespace:false,disableCombineTextItems:false});
        const pageText=extractPageText(tc.items, pageW, pageH);
        if(i===1){
          // Detectar columnas en la primera página para informar al usuario
          const body=tc.items.filter(it=>{
            const y=it.transform[5];
            return y>pageH*0.07&&y<pageH*0.93&&it.str.trim().length>3;
          });
          detectedCols=detectColumns(body, pageW);
        }
        pages.push(pageText);
      }
      text=pages.join('\n\n');
      const colLabel=detectedCols===1?'1 columna':`${detectedCols} columnas`;
      zone.querySelector('.uptxt').innerHTML=
        '<strong style="color:var(--acc)">'+file.name+'</strong><br>'
        +'<span style="font-size:.68rem;color:var(--mu)">'+_pageFormat
        +' · '+colLabel+' detectadas</span>';
    }
    else if(ext==='docx'){
      const buf=await file.arrayBuffer();
      _pageFormat=await detectFormatFromDocx(buf);
      const r=await mammoth.extractRawText({arrayBuffer:buf});text=r.value;
      zone.querySelector('.uptxt').innerHTML=
        '<strong style="color:var(--acc)">'+file.name+'</strong><br>'
        +'<span style="font-size:.68rem;color:var(--mu)">'+_pageFormat+' detectado</span>';
    }
    else if(ext==='doc'){
      // v7.8: .doc (Word 97-2003) requiere parser nativo backend porque
      // mammoth.js solo soporta .docx. Subimos al endpoint /api/extract-doc
      // que invoca antiword en el servidor. Ver backend/api/extract.py.
      zone.querySelector('.uptxt').innerHTML=
        '<strong style="color:var(--acc)">'+file.name+'</strong><br>'
        +'<span style="font-size:.68rem;color:var(--mu)">Convirtiendo en el servidor…</span>';
      const fd=new FormData(); fd.append('file', file);
      let resp;
      try {
        resp=await fetch(API+'/api/extract-doc',{method:'POST',body:fd});
      } catch(netErr) {
        throw new Error('No se pudo conectar al servidor para convertir el .doc. '
          +'Verifica tu conexion o convierte el archivo a .docx manualmente.');
      }
      // Fallback defensivo: backend desplegado sin el endpoint (deploy a medias)
      if(resp.status===404){
        throw new Error('Tu servidor no tiene soporte para .doc todavia (404). '
          +'Como workaround, abri el archivo en Word y guardalo como .docx, '
          +'o pega el texto directamente en el area de texto.');
      }
      if(!resp.ok){
        let detail='Error HTTP '+resp.status;
        try { const j=await resp.json(); if(j.detail) detail=j.detail; } catch(e){}
        throw new Error(detail);
      }
      const j=await resp.json();
      text=j.text;
      _pageFormat=j.format||'Carta';
      const partialNote=j.partial
        ? ' &middot; <span style="color:#F59E0B">recuperacion parcial (archivo danado)</span>'
        : '';
      zone.querySelector('.uptxt').innerHTML=
        '<strong style="color:var(--acc)">'+file.name+'</strong><br>'
        +'<span style="font-size:.68rem;color:var(--mu)">DOC convertido &middot; '
        +j.char_count+' chars'+partialNote+'</span>';
    }
    else{alert('Formato no soportado.');resetUpload();return;}

    document.getElementById('inp').value=text;
    // Solo actualizar el badge si no lo hicimos arriba (TXT y casos genéricos)
    if(ext==='txt'){
      zone.querySelector('.uptxt').innerHTML=
        '<strong style="color:var(--acc)">'+file.name+'</strong><br>'
        +'<span style="font-size:.68rem;color:var(--mu)">Carta (TXT) — analiza directamente</span>';
    }
  }catch(err){alert('Error al leer: '+err.message);resetUpload();}
}
function resetUpload(){
  document.getElementById('upz').querySelector('.uptxt').innerHTML=
    'Arrastra un archivo o <strong>haz clic</strong><br><span style="font-size:.68rem">TXT &middot; DOC &middot; DOCX &middot; PDF</span>';
  document.getElementById('fi').value='';
}
const upz=document.getElementById('upz');
upz.addEventListener('dragover',e=>{e.preventDefault();upz.style.borderColor='var(--acc)';});
upz.addEventListener('dragleave',()=>upz.style.borderColor='');
upz.addEventListener('drop',e=>{
  e.preventDefault();upz.style.borderColor='';
  const f=e.dataTransfer.files[0];
  if(f){document.getElementById('fi').files=e.dataTransfer.files;handleFile({target:{files:e.dataTransfer.files}});}
});
/* ── Lexicón compartido con el backend ──
   La fuente única de verdad vive en backend/core/lexicon.py y se expone vía
   GET /api/lexicon. Mientras la respuesta no llega (o si el backend está
   caído) se usa LEX_FALLBACK como única copia local de los patrones. */
// Helper: convierte "#2563EB" a "37,99,235" para uso en plantillas rgb()
function hexToRgb(hex){
  const m=String(hex||'').match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  return m ? `${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)}` : '128,128,128';
}

const LEX_FALLBACK={
  t1:{
    patterns:{
      INTRO:'\\b(objetivo|objetivos|introducci[oó]n|problema\\sde\\sinvestigaci[oó]n|justificaci[oó]n|prop[oó]sito|el\\spresente\\strabajo|esta\\sinvestigaci[oó]n|planteamiento)\\b',
      BACK: '\\b(antecedentes|estado\\sdel\\sarte|trabajos\\sprevios|marco\\ste[oó]rico|estudios\\sprevios|literatura|seg[uú]n|et\\sal|autores\\scomo)\\b',
      METH: '\\b(metodolog[íi]a|m[eé]todo|procedimiento|dise[nñ]o|muestra|participantes|cuestionario|an[aá]lisis\\sestad[íi]stico|tipo\\sde\\sestudio|muestreo)\\b',
      RES:  '\\b(resultados|hallazgos|los\\sresultados|se\\sencontr[oó]|se\\sobserv[oó]|estad[íi]sticamente|correlaci[oó]n|se\\sobtuvieron|en\\sla\\stabla|valor\\sde\\sp)\\b',
      DISC: '\\b(discusi[oó]n|estos\\sresultados\\ssugieren|esto\\simplica|estos\\shallazgos|en\\scontraste\\scon|coincide\\scon|difiere\\sde|lo\\scual\\sindica)\\b',
      CONTR:'\\b(contribuci[oó]n|nuestro\\senfoque|proponemos|presentamos|introducimos|nuestra\\spropuesta|a\\sdiferencia\\sde\\strabajos|se\\spropone\\sun\\snuevo|supera\\sal\\sestado|la\\saportaci[oó]n|aportaci[oó]n)\\b',
      LIM:  '\\b(limitaci[oó]n|limitaciones|no\\sfue\\sposible|sesgo|tama[nñ]o\\smuestral|muestra\\speque[nñ]a|futuras\\sinvestigaciones|trabajo\\sfuturo|restricci[oó]n)\\b',
      CONC: '\\b(conclusi[oó]n|conclusiones|en\\sconcluisi[oó]n|para\\sconcluir|se\\sconcluye\\sque|en\\sresumen|finalmente|se\\srecomienda)\\b',
    },
    priority:['CONTR','INTRO','CONC','LIM','METH','BACK','RES','DISC'],
    // Paleta de respaldo (idéntica a backend/core/lexicon.py).
    // El backend es la fuente de verdad: estos valores solo se usan si /api/lexicon falla.
    colors:{
      INTRO:'#2563EB', BACK:'#7C3AED', METH:'#84CC16', RES:'#16A34A',
      DISC:'#0F766E', CONTR:'#DC2626', LIM:'#C026D3', CONC:'#D97706',
    },
    names:{
      INTRO:'Introducción', BACK:'Antecedentes', METH:'Metodología',
      RES:'Resultados',     DISC:'Discusión',    CONTR:'Contribución',
      LIM:'Limitaciones',   CONC:'Conclusión',
    },
  },
  t2:{
    pattern:'\\b(proponemos|presentamos|introducimos|desarrollamos|contribuimos|aportamos|se\\spropone|se\\spresenta|se\\sintroduce|se\\sdesarrolla|se\\sconstruye|se\\screa|este\\strabajo\\sporta|nuestra\\spropuesta|nuestro\\sm[eé]todo|nuestro\\senfoque|nuestra\\sarquitectura|nuestro\\smodelo|nuestra\\ssoluci[oó]n|supera\\sal\\sestado\\sdel\\sarte|a\\sdiferencia\\sde\\strabajos|la\\sprincipal\\saportaci[oó]n|la\\sprincipal\\scontribuci[oó]n|contribuci[oó]n|aportaci[oó]n|el\\sobjetivo\\sde\\seste\\strabajo\\ses|ponemos\\sa\\sdisposici[oó]n|nuevo\\sm[eé]todo|nuevo\\senfoque|nueva\\sarquitectura|nueva\\smetodolog[íi]a)\\b',
  },
};

// Estado compilado en runtime — se rellena en compileLexicon()
let LEX={t1Patterns:{}, t1Priority:[], t1Meta:{}, t2Pattern:null};

function compileLexicon(payload){
  const src=payload||LEX_FALLBACK;
  LEX.t1Patterns=Object.fromEntries(
    Object.entries(src.t1.patterns).map(([k,p])=>[k,new RegExp(p,'gi')])
  );
  LEX.t1Priority=src.t1.priority||LEX_FALLBACK.t1.priority;
  // Construir el meta {rgb,n} desde colors/names que vienen del backend
  // (con fallback a la copia local si alguno faltara)
  const colors=src.t1.colors||LEX_FALLBACK.t1.colors;
  const names =src.t1.names ||LEX_FALLBACK.t1.names;
  LEX.t1Meta=Object.fromEntries(
    Object.keys(names).map(k=>[k,{rgb:hexToRgb(colors[k]),n:names[k]}])
  );
  LEX.t2Pattern=new RegExp(src.t2.pattern,'gi');
}
compileLexicon(LEX_FALLBACK); // listo desde el primer instante

async function loadLexicon(){
  try{
    const r=await fetch(API+'/api/lexicon',{signal:AbortSignal.timeout(3000)});
    if(!r.ok) throw new Error('http '+r.status);
    const payload=await r.json();
    compileLexicon(payload);
  }catch{
    // Backend no disponible — el fallback ya está compilado, no hacemos nada.
  }
}

/* ── Heurística T1 (usa LEX.t1Patterns) ── */
function t1classify(text,pos){
  const sc={};
  for(const l in LEX.t1Patterns){
    const re=LEX.t1Patterns[l];
    sc[l]=(text.match(re)||[]).length;
    re.lastIndex=0;
  }
  if(pos<=0.15){sc.INTRO+=3;sc.BACK+=2;}
  else if(pos>=0.80){sc.CONC+=3;sc.LIM+=2;}
  const best=LEX.t1Priority.reduce((a,b)=>sc[a]>=sc[b]?a:b);
  const tot=Object.values(sc).reduce((a,b)=>a+b,0)||1;
  const conf=sc[best]===0?0.42:Math.min(0.95,0.45+sc[best]/tot*0.5);
  const lbl=sc[best]===0?['INTRO','BACK','METH','RES','DISC','CONC'][Math.min(5,Math.floor(pos*6))]:best;
  return{lbl,conf:+conf.toFixed(3)};
}

/* ── Heurística T2 — independiente de T1 ── */
// T2 evalúa el CONTENIDO semántico del fragmento sin importar su categoría retórica.
// Una contribución puede declararse en CUALQUIER sección. T1 y T2 son ortogonales.
function t2classify(text){
  const re=LEX.t2Pattern;
  const ms=(text.match(re)||[]); re.lastIndex=0;
  const n=ms.length;
  if(n>=3)return{c:true, conf:Math.min(0.95,0.70+n*0.04)};
  if(n===2)return{c:true, conf:0.80};
  if(n===1)return{c:true, conf:0.65};
  return{c:false,conf:Math.max(0.48,0.84-text.split(/\s+/).length*0.0009)};
}
function splitP(text){
  // ── Paso 1: intentar con doble salto \n\n (estructura fuerte del documento)
  const byDouble = text.split(/\n{2,}/).map(p=>p.trim()).filter(p=>p.length>0);
  if(byDouble.length > 1){
    // Aun con dobles saltos, verificar si hay sub-fragmentos con \n simple
    // que sean encabezados de sección o filas de tabla (cortos, 1-5 palabras)
    const result = [];
    for(const block of byDouble){
      if(block.includes('\n')){
        // Dividir por \n simple dentro del bloque
        const lines = block.split('\n').map(l=>l.trim()).filter(l=>l.length>0);
        // Si hay líneas cortas al inicio (encabezado de sección), separarlas
        const firstShort = lines[0] && lines[0].split(/\s+/).length <= 6;
        if(firstShort && lines.length > 1){
          result.push(lines[0]);
          result.push(lines.slice(1).join(' '));
        } else {
          result.push(block.replace(/\n/g,' '));
        }
      } else {
        result.push(block);
      }
    }
    return result.filter(p=>p.length>0);
  }

  // ── Paso 2: intentar con salto simple \n (PDFs que no producen \n\n)
  const bySingle = text.split(/\n/).map(p=>p.trim()).filter(p=>p.length>0);
  if(bySingle.length > 3){
    // Agrupar líneas cortas como encabezados y concatenar hasta 25 palabras
    const groups = [], cur = [];
    let wc = 0;
    for(const line of bySingle){
      const lw = line.split(/\s+/).length;
      const isHeader = lw <= 5 && /^[A-ZÁÉÍÓÚÜÑ]/.test(line);
      // Separar encabezados como párrafo propio
      if(isHeader && cur.length > 0){
        groups.push(cur.join(' ')); cur.length=0; wc=0;
      }
      cur.push(line); wc+=lw;
      if(isHeader || wc >= 25){
        groups.push(cur.join(' ')); cur.length=0; wc=0;
      }
    }
    if(cur.length) groups.push(cur.join(' '));
    return groups.filter(p=>p.trim().length>0);
  }

  // ── Paso 3: texto sin estructura — dividir por oraciones en grupos de 25w
  const ss = text.split(/(?<=[.!?])\s+/);
  let g=[], w=0, gs=[];
  for(const s of ss){
    g.push(s); w+=s.split(/\s+/).length;
    if(w>=25){ gs.push(g.join(' ')); g=[]; w=0; }
  }
  if(g.length) gs.push(g.join(' '));
  return gs.filter(x=>x.trim().length>0).length ? gs.filter(x=>x.trim().length>0) : [text];
}

/* ══════════════════════════════════════════════════════════════════════════
   HEURÍSTICA LOCAL — dos funciones independientes (espejo del backend)
   ══════════════════════════════════════════════════════════════════════════ */

// T1 local: segmentación retórica
function localSegment(text, mid){
  const ps=splitP(text), n=ps.length;
  let charOffset=0;
  const segments=ps.map((p,i)=>{
    const pos=i/Math.max(n-1,1);
    const wordCount=p.split(/\s+/).filter(Boolean).length;
    const isShort=wordCount<4;
    const charStart=text.indexOf(p,charOffset);
    const charEnd=charStart>=0?charStart+p.length:charOffset+p.length;
    charOffset=charEnd;
    let lbl,conf;
    if(isShort){
      lbl=['INTRO','BACK','METH','RES','DISC','CONC'][Math.min(5,Math.floor(pos*6))];
      conf=0.30;
    }else{
      const r=t1classify(p,pos); lbl=r.lbl; conf=r.conf;
    }
    const zone=pos<=0.20?'Inicio (0-20%)':pos<=0.80?'Desarrollo (20-80%)':'Cierre (80-100%)';
    return{index:i, text:p,
      char_start:charStart, char_end:charEnd, word_count:wordCount,
      relative_pos:+pos.toFixed(4), rhetorical_zone:zone,
      t1_label:lbl, t1_label_name:LEX.t1Meta[lbl]?.n||lbl,
      t1_color:LEX.t1Meta[lbl]?`rgb(${LEX.t1Meta[lbl].rgb})`:'#9CA3AF',
      t1_confidence:+conf.toFixed(3), low_confidence:conf<=0.42, isShort};
  });
  const dist={};
  segments.filter(s=>s.t1_confidence>0.42).forEach(s=>{dist[s.t1_label]=(dist[s.t1_label]||0)+1;});
  const mname=mid==='heuristic'?'Heurístico (JS)':'Heurístico (fallback T1)';
  return{model_id:mid, model_name:mname, mode:'heuristic', elapsed_ms:3,
    segments,
    summary:{total_segments:n, t1_distribution:dist,
      avg_confidence:+(segments.reduce((a,s)=>a+s.t1_confidence,0)/Math.max(n,1)).toFixed(3),
      low_confidence_count:segments.filter(s=>s.low_confidence).length}};
}

// T2 local: detección de contribuciones (recibe segmentos de T1)
function localContributions(segments, mid){
  const results=segments.map(seg=>{
    const {c,conf:t2c}=t2classify(seg.text);
    const ctx=c?`Contribución en '${seg.t1_label_name}' (${(seg.t1_confidence*100).toFixed(0)}% conf. retórica) · ${seg.rhetorical_zone}`:'';
    return{index:seg.index, is_contribution:c, confidence:+t2c.toFixed(3),
      label:c?'ES_CONTRIBUCION':'NO_ES_CONTRIBUCION',
      t1_label:seg.t1_label, t1_label_name:seg.t1_label_name,
      t1_color:seg.t1_color, t1_confidence:seg.t1_confidence,
      rhetorical_zone:seg.rhetorical_zone, rhetorical_context:ctx,
      char_start:seg.char_start, char_end:seg.char_end,
      word_count:seg.word_count, relative_pos:seg.relative_pos,
      isShort:seg.isShort, text:seg.text};
  });
  const detected=results.filter(r=>r.is_contribution);
  const nc=detected.length;
  const byZone={}, byLabel={};
  detected.forEach(r=>{
    byZone[r.rhetorical_zone]=(byZone[r.rhetorical_zone]||0)+1;
    byLabel[r.t1_label]=(byLabel[r.t1_label]||0)+1;
  });
  const mname=(mid||'heuristic')==='heuristic'?'Heurístico (JS)':'Heurístico (fallback T2)';
  return{model_id:mid, model_name:mname, mode:'heuristic', elapsed_ms:2,
    contributions:results,
    summary:{total_fragments:results.length, contributions_detected:nc,
      contribution_rate:+(nc/Math.max(results.length,1)).toFixed(3),
      by_rhetorical_zone:byZone, by_t1_label:byLabel}};
}

// Combina los resultados T1+T2 en el formato interno del renderer
function mergeResults(segData, contribData){
  const contribMap={};
  contribData.contributions.forEach(c=>{contribMap[c.index]=c;});
  const paragraphs=segData.segments.map(s=>{
    const c=contribMap[s.index]||{is_contribution:false,confidence:0.5};
    const page=Math.ceil((s.index+1)/6);
    return{
      i:s.index, text:s.text,
      t1:s.t1_label, t1n:s.t1_label_name, t1c:s.t1_confidence,
      t2:c.is_contribution, t2c:c.confidence,
      isShort:s.isShort||s.low_confidence||false,
      charStart:s.char_start, charEnd:s.char_end,
      wordCount:s.word_count, posRel:s.relative_pos,
      zone:s.rhetorical_zone, page,
      rhetContext:c.rhetorical_context||'',
    };
  });
  const dist=segData.summary.t1_distribution||{};
  const nc=contribData.summary.contributions_detected;
  return{
    mname:segData.model_name, mname2:contribData.model_name,
    mode:segData.mode,
    ms:(segData.elapsed_ms||0)+(contribData.elapsed_ms||0),
    msT1:segData.elapsed_ms||0, msT2:contribData.elapsed_ms||0,
    paragraphs,
    segSummary:segData.summary,
    contribSummary:contribData.summary,
    sum:{n:paragraphs.length, nc,
      rate:contribData.summary.contribution_rate,
      dist, byZone:contribData.summary.by_rhetorical_zone||{},
      byLabel:contribData.summary.by_t1_label||{}},
  };
}

/* ── Analizar — dos llamadas al backend ── */
async function analyze(){
  const text=document.getElementById('inp').value.trim();
  if(!text){alert('Ingresa texto para analizar.');return;}

  // v7.4: si hay un analisis previo cargado, pedir confirmacion antes de
  // sobreescribirlo. Sin esto, el panel derecho muestra los resultados viejos
  // mientras el nuevo analisis esta en curso, lo que confunde al usuario.
  // _confirmClear() (en el modal) limpia el workspace antes de ejecutar la
  // accion pendiente, asi que el panel derecho queda vacio mientras corre.
  if(_needsConfirm()){
    _showClearModal('Hay un análisis previo cargado. Se descartará para realizar el nuevo análisis. ¿Continuar?',
                    () => _doAnalyze(text));
    return;
  }
  await _doAnalyze(text);
}

async function _doAnalyze(text){
  const btn=document.getElementById('abtn'),ic=document.getElementById('bico'),bt=document.getElementById('btxt');
  btn.disabled=true;ic.innerHTML='<div class="spin"></div>';bt.textContent='Analizando…';
  try{
    let data;
    try{
      // ── Paso 1: Tarea 1 — Segmentación retórica ──────────────────────────
      const r1=await fetch(API+'/api/segment',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({text, model_id:selModel})});
      if(!r1.ok) throw new Error(`T1 error: ${r1.status}`);
      const segData=await r1.json();

      // ── Paso 2: Tarea 2 — Detección de contribuciones ────────────────────
      // Enviamos los segmentos de T1 directamente (incluyen contexto retórico)
      const r2=await fetch(API+'/api/contributions',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({fragments:segData.segments, model_id:selModel2})});
      if(!r2.ok) throw new Error(`T2 error: ${r2.status}`);
      const contribData=await r2.json();

      data=mergeResults(segData,contribData);
    }catch(e){
      // Fallback: heurística local también en dos pasos
      console.info('Backend no disponible, usando heurística local');
      const segLocal=localSegment(text,selModel);
      const contribLocal=localContributions(segLocal.segments,selModel2);
      data=mergeResults(segLocal,contribLocal);
    }
    render(data);
  }catch(e){alert('Error: '+e.message);}
  finally{btn.disabled=false;ic.textContent='▶';bt.textContent='Analizar';}
}
/* ── Render ── */
const PPG=6;
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function render(d){
  document.getElementById('es').style.display='none';
  const res=document.getElementById('res');res.style.display='flex';
  _lastData=d;
  const s=d.sum;
  // Meta strip — modelos T1 y T2 con tiempos independientes
  const totalWords=d.paragraphs.reduce((acc,p)=>acc+(p.wordCount||p.text.split(/\s+/).filter(Boolean).length),0);
  const t1ms=d.msT1||d.ms||0, t2ms=d.msT2||0;
  document.getElementById('meta').innerHTML=
    `<span>Detección Retórica: <span class="mv">${d.mname}</span></span>
     <span>Detección de Contribuciones: <span class="mv">${d.mname2||d.mname}</span></span>
     <span>Párrafos: <span class="mv">${s.n}</span></span>
     <span>Palabras: <span class="mv">${totalWords.toLocaleString('es-CO')}</span></span>
     <span style="margin-left:auto">⏱ T1: <span class="mv">${t1ms} ms</span> · T2: <span class="mv">${t2ms} ms</span></span>`;
  // Dist KPIs — T1 distribuye 100% entre 7 etiquetas (sin CONTR).
  // CONTR es ortogonal: su valor viene de T2 (s.nc = contribuciones detectadas)
  // y su % es sobre el total de párrafos del documento (no sobre el % T1).
  // Hacer doble-contar CONTR contra una etiqueta T1 inflaría >100%, así que
  // CONTR queda fuera de la repartición pero presente como KPI propio.
  const dist=s.dist;
  const nLow=d.paragraphs.filter(p=>p.isShort||(p.t1c<=0.42)).length;
  const nAll=d.paragraphs.length||1;
  const keys8=Object.keys(LEX.t1Meta);
  const t1Keys=keys8.filter(k=>k!=='CONTR');                  // 7 etiquetas T1
  const t1Total=t1Keys.reduce((a,k)=>a+(dist[k]||0),0)||1;
  // Distribuir floors+remainder solo sobre t1Keys (CONTR se calcula aparte)
  const rawsT1=t1Keys.map(k=>(dist[k]||0)/t1Total*100);
  const floorsT1=rawsT1.map(r=>Math.floor(r));
  const floorSum=floorsT1.reduce((a,b)=>a+b,0);
  const remainder=100-floorSum;
  const fracs=rawsT1.map((r,i)=>({i,frac:r-floorsT1[i]})).sort((a,b)=>b.frac-a.frac);
  const pctsT1=[...floorsT1];
  for(let j=0;j<remainder;j++) pctsT1[fracs[j].i]++;
  // Mapa label → pct para lookup en el render
  const pctMap=Object.fromEntries(t1Keys.map((k,i)=>[k, pctsT1[i]]));
  // CONTR: su conteo es s.nc (T2) y su pct es sobre total párrafos
  const ncFromT2 = s.nc || 0;
  pctMap['CONTR'] = Math.round(ncFromT2 / nAll * 100);

  document.getElementById('dcard').innerHTML=
    `<div class="dtit">Distribución Retórica</div>
     <div class="drows">
       ${keys8.map((k)=>{
         const v=LEX.t1Meta[k];
         // CONTR usa conteo T2; las demás usan dist T1
         const cnt = k==='CONTR' ? ncFromT2 : (dist[k]||0);
         const pct = pctMap[k];
         const dim=cnt===0?'opacity:.35':'';
         const titleSuffix = k==='CONTR'
           ? 'detectadas por T2 — clic para filtrar contribuciones'
           : `clic para filtrar por ${v.n}`;
         return `<div class="drow" data-cat="${k}" onclick="toggleKpiFilter('${k}')"
           style="border-top:3px solid rgb(${v.rgb});${dim};cursor:pointer"
           title="${titleSuffix}">
           <div class="dval" style="color:rgb(${v.rgb})">${cnt}</div>
           <div class="dpct">${pct}%</div>
           <div class="dlbl">
             <span class="dlbl-dot" style="background:rgb(${v.rgb})"></span>
             <span>${k}</span>
           </div>
         </div>`;
       }).join('')}
       ${nLow>0?`<div class="drow" data-cat="__low" onclick="toggleKpiFilter(null)"
         style="border-top:3px solid #9CA3AF;opacity:.7;cursor:pointer"
         title="Clic para resetear el filtro">
         <div class="dval" style="color:#9CA3AF">${nLow}</div>
         <div class="dpct">${Math.round(nLow/nAll*100)}%</div>
         <div class="dlbl">
           <span class="dlbl-dot" style="background:#9CA3AF"></span>
           <span style="font-style:italic">Sin cat.</span>
         </div>
       </div>`:''}
     </div>`;
  // Resetear filtro al cargar nuevo análisis
  _kpiFilter.clear();
  // Páginas — pinta TODO el documento (cada párrafo, con o sin categoría detectada)
  const paras=d.paragraphs,total=Math.ceil(paras.length/PPG);
  const pw=document.getElementById('pages');
  pw.innerHTML='';
  pw.dataset.fmt=_pageFormat;
  for(let pg=0;pg<total;pg++){
    const sl=paras.slice(pg*PPG,(pg+1)*PPG);
    const div=document.createElement('div');div.className='ppage';
    div.innerHTML=`<div class="phdr">
        <span>MAIA · Análisis Retórico y de Contribuciones</span>
        <span style="display:flex;align-items:center;gap:.5rem">
          <span class="pfmt">${_pageFormat}</span>
          <span>${d.mname}</span>
        </span>
      </div>
      ${sl.map(p=>renderP(p)).join('')}
      <div class="pnum">Página ${pg+1} de ${total}</div>`;
    pw.appendChild(div);
  }
  renderExport(d);
}
/* ── Renderizado de párrafos ── */
// Divide texto en oraciones para resaltar solo las que contienen contribución
function splitSentences(text){
  // Divide en oraciones preservando puntuación
  return text.match(/[^.!?]+[.!?]*/g)||[text];
}

function hasCONTR(sent){
  // Reutiliza el patrón T2 ya compilado desde el lexicón compartido
  const r=LEX.t2Pattern;
  const result=r.test(sent); r.lastIndex=0; return result;
}

function renderSentences(text, t2pct){
  // Resalta oraciones con contribución y añade tooltip de confianza T2
  // sobre la propia oración resaltada (no en el tooltip del párrafo).
  //
  // v7 fix paridad visual<->CSV: el modelo T2 (encoder SciBETO o LLM) puede
  // marcar p.t2=true a nivel de párrafo basándose en señales semánticas que
  // NO coinciden con el patrón léxico simple `LEX.t2Pattern` usado aquí para
  // granular el resaltado a nivel de oración. Cuando eso ocurre, el CSV
  // (que se basa en p.t2) reporta la contribución pero en pantalla no se ve
  // ningún resaltado. Solución: si ninguna sub-oración matchea el patrón,
  // resaltar el párrafo COMPLETO como contribución (caso típico: el párrafo
  // es una sola oración corta, o la "contribución" cubre todo el párrafo).
  //
  // v7.7: si t2pct es null el modelo T2 es un LLM (no devuelve prob real),
  // omitimos el segmento "· confianza T2: NN%" del tooltip.
  const confInfo = (t2pct===null||t2pct===undefined) ? '' : ` · confianza T2: ${t2pct}%`;
  const sents=splitSentences(text);
  const matches=sents.map(s=>hasCONTR(s));
  const anyMatch=matches.some(Boolean);

  if(!anyMatch){
    // Fallback: resaltar el párrafo entero. Mantiene la misma clase y tooltip
    // que el highlight oración-a-oración, así el usuario tiene UX consistente
    // y NO se pierde ninguna contribución reportada en el CSV.
    return `<span class="contr-sent">
      <span class="contr-tip">ES_CONTRIBUCION${confInfo} · resaltado a nivel de parrafo</span>${esc(text)}</span>`;
  }

  return sents.map((s,i)=>{
    if(matches[i]){
      return `<span class="contr-sent">
        <span class="contr-tip">ES_CONTRIBUCION${confInfo}</span>${esc(s)}</span>`;
    }
    return esc(s);
  }).join('');
}

function renderP(p){
  const m=LEX.t1Meta[p.t1]||{rgb:'128,128,128',n:p.t1};
  const LOW_CONF = p.t1c <= 0.42;
  const SHORT    = p.isShort;
  const rgb = (LOW_CONF||SHORT) ? '160,160,160' : m.rgb;
  const isC = p.t2;
  const t1pct=(p.t1c*100).toFixed(0);
  // v7.7: si T1 o T2 son LLM, NO mostramos % en los tooltips (LLMs greedy
  // no devuelven probs calibradas — la "confidence" es siempre 100% en
  // backend, lo cual confunde). SciBETO y heuristico si tienen probs.
  const t1IsLLM = _isLLM(selModel);
  const t2IsLLM = _isLLM(selModel2);
  const t2pctOrNull = t2IsLLM ? null : (p.t2c*100).toFixed(0);
  const bg = `rgba(${rgb},var(--ha))`;
  const bl = `rgb(${rgb})`;

  // Contar cuantas oraciones tienen marcadores de contribucion segun el
  // patron lexico local. Si el modelo marco isC=true pero ninguna oracion
  // matchea (caso semantico que el patron lexico no cubre), tratamos el
  // parrafo entero como UNA contribucion -- consistente con el resaltado a
  // nivel de parrafo en renderSentences y con la fila del CSV.
  let nContrSents = 0;
  let paragraphLevel = false;
  if(isC){
    const sents=splitSentences(p.text);
    nContrSents=sents.filter(s=>hasCONTR(s)).length;
    if(nContrSents===0){
      nContrSents=1;
      paragraphLevel=true;
    }
  }
  const contrLabel = paragraphLevel
    ? '1 contribución (parrafo completo)'
    : (nContrSents === 1 ? '1 contribución' : `${nContrSents} contribuciones`);

  // Cuerpo: confianza T2 va en title de cada oración resaltada, NO en el tooltip del párrafo
  const body = isC ? renderSentences(p.text, t2pctOrNull) : esc(p.text);

  // Etiqueta tooltip T1
  const tipLabel = SHORT
    ? 'Fragmento corto'
    : LOW_CONF
      ? `Posible ${m.n}`
      : m.n;

  const _dc=(LOW_CONF||SHORT)?"__low":p.t1;
  return `<div class="ap" data-t1="${_dc}" data-t2="${isC ? 'true' : 'false'}"
    style="background:${bg};border-left-color:${bl}">
    <div class="tip">
      <span class="bt1" style="background:rgba(${rgb},.2);color:rgb(${rgb});
        border:1px solid rgba(${rgb},var(--hb));${LOW_CONF||SHORT?'font-style:italic':''}">
        ${tipLabel}
      </span>
      ${(!SHORT && !t1IsLLM)
        ?`<div class="cm">
            <div class="ctrk"><div class="cfil" style="width:${t1pct}%;background:rgb(${rgb})"></div></div>
            <span style="font-size:.67rem;color:var(--mu)">${t1pct}%</span>
          </div>`:''}
      ${isC
        ?`<span class="bt2y">${contrLabel}</span>`
        :`<span class="bt2n">sin contribución</span>`}
    </div>
    ${body}
  </div>`;
}

/* ── Modal About us ── */
function openAbout(){
  document.getElementById('aboutModal').classList.add('open');
}
function closeAbout(){
  document.getElementById('aboutModal').classList.remove('open');
}
/* ── Modal de ayuda ── */
function openHelp(){
  // Poblar leyenda dentro del modal (siempre regenera, así refleja LEX.t1Meta actualizado)
  const leg=document.getElementById('hLegend');
  if(leg){
    leg.innerHTML=Object.entries(LEX.t1Meta).map(([k,v])=>`
      <div style="display:flex;align-items:center;gap:.5rem;
           background:rgba(${v.rgb},.1);border:1px solid rgba(${v.rgb},.25);
           border-radius:7px;padding:.4rem .7rem">
        <span style="width:10px;height:10px;border-radius:2px;background:rgb(${v.rgb});flex-shrink:0"></span>
        <span style="font-size:.8rem;font-weight:500">${v.n}</span>
        <span style="font-size:.72rem;color:var(--mu);margin-left:auto">${k}</span>
      </div>`).join('');
  }
  const m=document.getElementById('helpModal');
  m.classList.add('open');
}
function closeHelp(){
  document.getElementById('helpModal').classList.remove('open');
}
/* ── Modal de few-shots (v7) ──
   Consume GET /api/few_shots para mostrar los ejemplos publicados de T1 (k=14)
   y T2 (k=8) con tabs. Single source of truth: backend/core/llm_service.py.
   El cache es por sesion (window._fsCache) para evitar refetch al re-abrir. */
let _fsActiveTab='T1';

async function openFewshot(){
  const m=document.getElementById('fewshotModal');
  m.classList.add('open');
  // Resetear a tab T1 al abrir
  _fsActiveTab='T1';
  document.getElementById('fsTabT1').classList.add('fs-tab-active');
  document.getElementById('fsTabT2').classList.remove('fs-tab-active');

  if(!window._fsCache){
    try{
      const r=await fetch(API+'/api/few_shots');
      if(!r.ok) throw new Error('HTTP '+r.status);
      window._fsCache=await r.json();
    }catch(e){
      document.getElementById('fsMeta').textContent='';
      document.getElementById('fsList').innerHTML=`
        <div style="text-align:center;padding:2rem;color:#EF4444;font-size:.85rem">
          No se pudieron cargar los ejemplos: ${esc(String(e.message||e))}.<br>
          <span style="color:var(--mu);font-size:.78rem">
            Verifica que el backend este activo. Endpoint: GET /api/few_shots</span>
        </div>`;
      return;
    }
  }
  _renderFewshotTab(_fsActiveTab);
}

function closeFewshot(){
  document.getElementById('fewshotModal').classList.remove('open');
}

function switchFewshotTab(task){
  _fsActiveTab=task;
  document.getElementById('fsTabT1').classList.toggle('fs-tab-active', task==='T1');
  document.getElementById('fsTabT2').classList.toggle('fs-tab-active', task==='T2');
  _renderFewshotTab(task);
}

function _renderFewshotTab(task){
  if(!window._fsCache) return;
  const d=window._fsCache[task];
  if(!d) return;

  // Colores por label: T1 reutiliza LEX.t1Meta (azul intro, lila back, etc),
  // T2 distingue SI/NO con verde y gris (no rojo: aca el rojo se reserva para
  // resaltado de contribuciones en pantalla).
  const colorFor=(task==='T1')
    ? lbl=>{const m=LEX.t1Meta[lbl];return m?m.rgb:'128,128,128';}
    : lbl=>lbl==='SI'?'16,185,129':'107,114,128';

  // Header meta (k, balance, orden, SHA truncado)
  document.getElementById('fsMeta').innerHTML=`
    <div style="display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;font-size:.77rem">
      <div style="color:var(--mu)">Tarea:</div>
      <div style="color:var(--tx)"><strong>${esc(d.task)}</strong></div>
      <div style="color:var(--mu)">Total ejemplos (k):</div>
      <div style="color:var(--tx)"><strong>${d.k}</strong></div>
      <div style="color:var(--mu)">Balance:</div>
      <div style="color:var(--tx)">${esc(d.balance)}</div>
      <div style="color:var(--mu)">Orden:</div>
      <div style="color:var(--tx)">${esc(d.order)}</div>
      <div style="color:var(--mu)">SHA256:</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                  color:var(--acc);word-break:break-all">${esc(d.sha256)}</div>
    </div>`;

  // Lista de ejemplos
  document.getElementById('fsList').innerHTML=d.examples.map((ex,i)=>{
    const rgb=colorFor(ex.label);
    return `
      <div style="border:1px solid var(--brd);border-radius:8px;overflow:hidden">
        <div style="display:flex;align-items:center;justify-content:space-between;
                    background:rgba(${rgb},.1);padding:.45rem .8rem;
                    border-bottom:1px solid rgba(${rgb},.25)">
          <div style="display:flex;align-items:center;gap:.55rem">
            <span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                         color:var(--mu);min-width:1.4rem">#${i+1}</span>
            <span style="background:rgb(${rgb});color:#fff;border-radius:4px;
                         padding:.12rem .5rem;font-size:.7rem;font-weight:700;
                         font-family:'JetBrains Mono',monospace">${esc(ex.label)}</span>
          </div>
        </div>
        <div style="padding:.55rem .9rem;font-size:.81rem;line-height:1.55;
                    color:var(--tx);background:var(--surf)">${esc(ex.text)}</div>
      </div>`;
  }).join('');
}
/* ── Modal de confirmación — limpiar workspace ── */
let _pendingAction = null;  // función a ejecutar tras confirmar

function _needsConfirm(){
  // Solo pedir confirmación si ya hay análisis cargado
  return _lastData !== null;
}

function _showClearModal(msg, action){
  document.getElementById('clearModalMsg').textContent = msg;
  _pendingAction = action;
  const m = document.getElementById('clearModal');
  m.style.display = 'flex';
}

function _confirmClear(){
  document.getElementById('clearModal').style.display = 'none';
  // Limpiar workspace
  _lastData = null;
  _kpiFilter.clear();
  document.getElementById('es').style.display = '';
  document.getElementById('res').style.display = 'none';
  document.getElementById('pages').innerHTML = '';
  if(_pendingAction){ _pendingAction(); _pendingAction = null; }
}

function _cancelClear(){
  document.getElementById('clearModal').style.display = 'none';
  _pendingAction = null;
}

function loadExample(){
  const doLoad = () => {
    document.getElementById('inp').value=`El análisis automático de artículos científicos representa un desafío central en el procesamiento del lenguaje natural. El presente trabajo tiene como objetivo desarrollar un sistema de clasificación retórica para documentos científicos en español.

Los antecedentes en este campo se remontan a los trabajos de Swales (1990) sobre la estructura retórica. Estudios previos han abordado la segmentación en inglés. La literatura científica en español ha recibido menos atención.

La metodología consiste en la construcción de un corpus de 1.8 millones de documentos de la plataforma CORE. Se aplicó un pipeline con ventana deslizante de 250 a 1000 palabras. El diseño metodológico es de tipo cuantitativo con análisis estadístico multivariado.

Los resultados muestran que el 37% de las contribuciones aparecen en el primer 15% del documento. Se encontró que el clasificador alcanzó una precisión del 94%. La correlación entre posición y tipo retórico fue estadísticamente significativa.

En la discusión, estos hallazgos sugieren que los patrones léxico-sintácticos son señales efectivas. Nuestro enfoque puede aprender la estructura retórica sin anotación exhaustiva. A diferencia de trabajos previos, nuestra solución no requiere etiquetado manual de todo el corpus.

Este trabajo propone un nuevo enfoque para la extracción automática de contribuciones que supera al estado del arte. Proponemos una arquitectura modular que integra segmentación retórica con detección binaria. La principal contribución es un pipeline reproducible y eficiente para el español académico.

Entre las limitaciones se encuentran la dependencia del corpus CORE y el reducido tamaño muestral. Futuras investigaciones deberán ampliar el conjunto de evaluación.

En conclusión, los resultados confirman la viabilidad del enfoque propuesto. Este estudio ha demostrado que los patrones léxico-sintácticos son señales robustas. Se recomienda ampliar el corpus en trabajos futuros.`;
  };
  if(_needsConfirm()){
    _showClearModal('Se borrará el análisis actual para cargar el texto de ejemplo. ¿Continuar?', doLoad);
  } else {
    doLoad();
  }
}

/* ── Exportación ── */
let _lastData=null;
// Filtro activo de categorías retóricas. null = sin filtro (todo visible)
let _kpiFilter=new Set(); // vacío = mostrar todas las categorías

function _filterLabel(){
  if(_kpiFilter.size===0) return 'ALL';
  // Orden canónico para el nombre de archivo
  const order=['INTRO','BACK','METH','RES','DISC','CONTR','LIM','CONC'];
  return order.filter(k=>_kpiFilter.has(k)).join('_');
}

function toggleKpiFilter(cat){
  if(!cat){_kpiFilter.clear();}         // clic en "Sin cat." → reset total
  else if(_kpiFilter.has(cat)) _kpiFilter.delete(cat);
  else _kpiFilter.add(cat);
  _applyFilter();
}

// Regla única de filtrado compartida entre _applyFilter (vista) y exports.
// - sin filtro → todo pasa
// - filtro incluye CONTR → el párrafo pasa si t2=true (es contribución según T2)
// - filtro incluye etiqueta T1 → el párrafo pasa si su T1 está en el filtro
//   con alta confianza (t1c > 0.42)
// La regla es OR entre criterios: filtro=[INTRO, CONTR] muestra todos los INTRO
// más todas las contribuciones (no la intersección).
function _paragraphMatchesFilter(p){
  if(_kpiFilter.size === 0) return true;
  const wantsContr = _kpiFilter.has('CONTR');
  if(wantsContr && p.t2 === true) return true;
  // T1 filter: solo aplica a etiquetas T1 reales (no CONTR), y solo a párrafos
  // con confianza alta — los low-confidence quedan fuera para no contaminar la vista.
  if(p.t1c > 0.42 && _kpiFilter.has(p.t1)) return true;
  return false;
}

function _applyFilter(){
  const hasFilter = _kpiFilter.size > 0;
  // Actualizar aspecto de cada KPI
  document.querySelectorAll('.drow[data-cat]').forEach(el=>{
    const cat = el.dataset.cat;
    if(cat === '__low'){
      // KPI "Sin cat." se atenúa siempre que haya filtro activo
      el.classList.toggle('kpi-inactive', hasFilter);
    } else {
      el.classList.toggle('kpi-inactive', hasFilter && !_kpiFilter.has(cat));
    }
  });
  // Actualizar visibilidad de los párrafos — low-confidence SIEMPRE se atenúan con filtro
  document.querySelectorAll('.ap[data-t1]').forEach(el=>{
    const cat = el.dataset.t1;
    const isLow = cat === '__low';
    const isContr = el.dataset.t2 === 'true';
    let show;
    if(!hasFilter){
      show = true;
    } else if(_kpiFilter.has('CONTR') && isContr){
      // Filtro de CONTR alcanza a cualquier párrafo marcado como contribución por T2
      show = true;
    } else if(isLow){
      show = false;
    } else {
      show = _kpiFilter.has(cat);
    }
    el.style.opacity = show ? '1' : '0.08';
    el.style.transition = 'opacity .25s';
  });
  // Hint de filtro
  const hint = document.getElementById('filter-hint');
  if(hint){
    hint.textContent = !hasFilter
      ? 'Haz clic en un KPI para filtrar'
      : `Filtrando: ${[..._kpiFilter].join(', ')} — clic de nuevo para quitar · clic en Sin cat. para ver todo`;
  }
}

function renderExport(d){
  document.getElementById('expcard').innerHTML=`
    <div style="display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap">
      <div class="exptit" style="margin:0">Exportar resultados</div>
      <div id="filter-hint" style="font-size:.69rem;color:var(--mu)">Haz clic en un KPI para filtrar</div>
    </div>
    <div class="expbtns" style="margin-top:.6rem">
      <button class="expbtn" onclick="exportJSON()">📄 JSON</button>
      <button class="expbtn" onclick="exportCSV()">📊 CSV retórico</button>
      <button class="expbtn" onclick="exportContribCSV()">🎯 CSV contribuciones</button>
      <button class="expbtn" onclick="exportHTML()">🌐 HTML</button>
      <button class="expbtn" onclick="exportPDF()">🖨 PDF</button>
    </div>`;
}

function download(name,content,type){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([content],{type}));
  a.download=name; a.click(); URL.revokeObjectURL(a.href);
}

function exportJSON(){
  if(!_lastData)return;
  download('maia_resultados.json',JSON.stringify(_lastData,null,2),'application/json');
}

function exportCSV(){
  if(!_lastData)return;
  const bom='\uFEFF';
  const rows=[['#','T1_etiqueta','T1_nombre','T1_confianza','confianza_baja',
               'ES_CONTRIBUCION','T2_confianza','zona_retorica','posicion_relativa',
               'pagina_estimada','palabras','texto']];
  _lastData.paragraphs.forEach(p=>{
    rows.push([
      p.i+1, p.t1, p.t1n||p.t1, p.t1c,
      (p.t1c<=0.42?'SI':'NO'),
      p.t2?'ES_CONTRIBUCION':'NO_ES_CONTRIBUCION', p.t2c,
      p.zone||'', p.posRel||'',
      p.page||'', p.wordCount||'',
      `"${(p.text||'').replace(/"/g,'""')}"`,
    ]);
  });
  download('maia_retorico.csv', bom+rows.map(r=>r.join(',')).join('\n'),'text/csv;charset=utf-8');
}

function exportContribCSV(){
  if(!_lastData)return;
  const bom='\uFEFF';
  const contribs=_lastData.paragraphs.filter(p=>p.t2);
  if(!contribs.length){alert('No se detectaron contribuciones en el texto.');return;}
  const rows=[['#','contexto_retorico','confianza_t2','zona','posicion_relativa',
               'pagina_estimada','palabras','char_inicio','char_fin','fragmento']];
  contribs.forEach((p,idx)=>{
    rows.push([
      idx+1,
      `${p.t1n||p.t1} (${(p.t1c*100).toFixed(0)}%)`,
      (p.t2c*100).toFixed(0)+'%',
      p.zone||'', p.posRel||'',
      p.page||'', p.wordCount||'',
      p.charStart!=null?p.charStart:'',
      p.charEnd!=null?p.charEnd:'',
      `"${(p.text||'').replace(/"/g,'""')}"`,
    ]);
  });
  download('maia_contribuciones.csv', bom+rows.map(r=>r.join(',')).join('\n'),'text/csv;charset=utf-8');
}

function exportHTML(){
  if(!_lastData)return;
  const d=_lastData;
  const T1HEX={INTRO:'2563EB',BACK:'7C3AED',METH:'84CC16',RES:'16A34A',DISC:'0F766E',CONTR:'DC2626',LIM:'C026D3',CONC:'D97706'};
  const srcParas=d.paragraphs.filter(_paragraphMatchesFilter);
  const paras=srcParas.map(p=>{
    const hex=p.t1c<=0.42?'9CA3AF':(T1HEX[p.t1]||'9CA3AF');
    const rgb=`${parseInt(hex.slice(0,2),16)},${parseInt(hex.slice(2,4),16)},${parseInt(hex.slice(4,6),16)}`;
    const sents=splitSentences(p.text);
    const body=p.t2
      ?sents.map(s=>hasCONTR(s)
        ?`<span style="background:rgba(239,68,68,.22);border-radius:3px;padding:0 2px;
            border-bottom:1.5px solid rgba(239,68,68,.6)">${s.replace(/</g,'&lt;')}</span>`
        :s.replace(/</g,'&lt;')).join('')
      :p.text.replace(/</g,'&lt;');
    // T1: etiqueta retórica + confianza (solo si no es posicional)
    const t1label=p.t1c<=0.42?`Posible ${LEX.t1Meta[p.t1]?.n||p.t1}`:(LEX.t1Meta[p.t1]?.n||p.t1);
    const t1conf=p.t1c>0.42?` · ${(p.t1c*100).toFixed(0)}%`:'';
    // T2: cantidad de contribuciones + confianza (solo si hay contribución)
    const nC=p.t2?sents.filter(s=>hasCONTR(s)).length:0;
    const t2label=p.t2
      ?` &nbsp;<span style="color:#EF4444;font-style:normal;font-weight:700">`
       +`${nC===1?'1 contribución':`${nC} contribuciones`}`
       +`</span>`
      :'';
    return `<div style="margin-bottom:.8rem;padding:.6rem .85rem;border-radius:5px;
      border-left:3px solid #${hex};background:rgba(${rgb},.1);
      font-size:14px;line-height:1.75;font-family:Georgia,serif">
      <div style="font-size:10.5px;margin-bottom:.3rem;color:#${hex};font-weight:600;
           font-family:sans-serif;font-style:${p.t1c<=0.42?'italic':'normal'}">
        ${t1label}${t1conf}${t2label}
      </div>${body}</div>`;
  }).join('');
  const html=`<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
    <title>MAIA · Informe de análisis</title>
    <style>
    body{max-width:860px;margin:2rem auto;font-family:sans-serif;background:#f8fafc;color:#1a2233;padding:1rem 2rem}
    h1{font-size:1.4rem;margin-bottom:.3rem} h2{font-size:.95rem;color:#4b5563;margin:.2rem 0 1rem}
    .meta{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:.7rem 1rem;
      margin-bottom:1.2rem;font-size:.82rem;color:#6b7280;display:flex;gap:1.5rem;flex-wrap:wrap}
    .mv{font-weight:600;color:#1a2233}
    </style></head><body>
    <h1>MAIA · Análisis Retórico y de Contribuciones</h1>
    <div class="meta">
      <span>T1: <span class="mv">${d.mname}</span></span>
      <span>T2: <span class="mv">${d.mname2||d.mname}</span></span>
      <span>Párrafos: <span class="mv">${d.sum.n}</span></span>
      <span>Contribuciones: <span class="mv">${d.sum.nc}</span></span>
    </div>
    ${paras}
    <p style="font-size:.75rem;color:#9ca3af;margin-top:2rem">
      Generado por MAIA · ${new Date().toLocaleString('es-CO')}</p>
    </body></html>`;
  download(`rethorical_${_filterLabel()}.html`,html,'text/html');
}

function exportPDF(){
  if(!_lastData)return;
  const d=_lastData;
  const T1HEX={INTRO:'2563EB',BACK:'7C3AED',METH:'84CC16',RES:'16A34A',DISC:'0F766E',CONTR:'DC2626',LIM:'C026D3',CONC:'D97706'};
  // Construimos un HTML optimizado para impresión y abrimos la ventana de impresión
  // El usuario elige "Guardar como PDF" desde el diálogo del navegador.
  const srcParas2=d.paragraphs.filter(_paragraphMatchesFilter);
  const paras=srcParas2.map(p=>{
    const hex=p.t1c<=0.42?'9CA3AF':(T1HEX[p.t1]||'9CA3AF');
    const rgb=`${parseInt(hex.slice(0,2),16)},${parseInt(hex.slice(2,4),16)},${parseInt(hex.slice(4,6),16)}`;
    const sents=splitSentences(p.text);
    const body=p.t2
      ?sents.map(s=>hasCONTR(s)
        ?`<span style="-webkit-print-color-adjust:exact;print-color-adjust:exact;
            background:rgba(239,68,68,.28);border-radius:2px;padding:0 2px;
            border-bottom:1.5px solid rgba(239,68,68,.7)">${s.replace(/</g,'&lt;')}</span>`
        :s.replace(/</g,'&lt;')).join('')
      :p.text.replace(/</g,'&lt;');
    const t1label=p.t1c<=0.42?`Posible ${LEX.t1Meta[p.t1]?.n||p.t1}`:(LEX.t1Meta[p.t1]?.n||p.t1);
    const t1conf=p.t1c>0.42?` · ${(p.t1c*100).toFixed(0)}%`:'';
    const nC=p.t2?sents.filter(s=>hasCONTR(s)).length:0;
    const t2label=p.t2
      ?` | <span style="color:#DC2626;font-weight:700">`
       +`${nC===1?'1 contribución':`${nC} contribuciones`}`
       +`</span>`
      :'';
    return `<div style="-webkit-print-color-adjust:exact;print-color-adjust:exact;
      margin-bottom:.6rem;padding:.5rem .75rem;border-radius:4px;
      border-left:3px solid #${hex};background:rgba(${rgb},.1);
      font-size:11.5pt;line-height:1.7">
      <div style="font-size:8pt;margin-bottom:.25rem;color:#${hex};font-weight:700;
           font-style:${p.t1c<=0.42?'italic':'normal'}">
        ${t1label}${t1conf}${t2label}
      </div>${body}</div>`;
  }).join('');
  // Leyenda para el PDF
  const leyenda=Object.entries(T1HEX).map(([k,hex])=>`
    <span style="display:inline-flex;align-items:center;gap:3px;margin-right:8px;font-size:7.5pt">
      <span style="width:8px;height:8px;border-radius:2px;background:#${hex};
        display:inline-block;-webkit-print-color-adjust:exact;print-color-adjust:exact"></span>
      ${LEX.t1Meta[k]?.n||k}
    </span>`).join('');
  const win=window.open('','_blank');
  // nombre archivo PDF
  const _pdfName=`rethorical_${_filterLabel()}.pdf`;
  win.document.title=_pdfName;
  win.document.write(`<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
    <title>MAIA · Informe PDF</title>
    <style>
    @page{size:Letter;margin:2cm 2.2cm}
    body{font-family:'Georgia',serif;color:#1a2233;font-size:11.5pt;line-height:1.7;margin:0}
    .header{border-bottom:2px solid #1F4E79;padding-bottom:.4rem;margin-bottom:1rem}
    h1{font-size:14pt;color:#1F4E79;margin:0 0 .15rem}
    .meta{font-size:8pt;color:#6b7280;display:flex;gap:1rem;flex-wrap:wrap}
    .mv{font-weight:700;color:#1a2233}
    .leyenda{margin:.5rem 0 1rem;padding:.35rem .5rem;border:1px solid #e5e7eb;
      border-radius:4px;font-size:7.5pt;background:#fafafa;
      -webkit-print-color-adjust:exact;print-color-adjust:exact}
    .footer{font-size:7pt;color:#9ca3af;margin-top:1.5rem;text-align:center;
      border-top:1px solid #e5e7eb;padding-top:.4rem}
    </style></head><body>
    <div class="header">
      <h1>MAIA · Análisis Retórico y de Contribuciones</h1>
      <div class="meta">
        <span>T1: <span class="mv">${d.mname}</span></span>
        <span>T2: <span class="mv">${d.mname2||d.mname}</span></span>
        <span>Párrafos: <span class="mv">${d.sum.n}</span></span>
        <span>Contribuciones: <span class="mv">${d.sum.nc}</span></span>
      </div>
    </div>
    <div class="leyenda">Leyenda retórica: ${leyenda}
      &nbsp;·&nbsp;
      <span style="display:inline-flex;align-items:center;gap:3px">
        <span style="width:8px;height:8px;border-radius:50%;background:#DC2626;
          display:inline-block;-webkit-print-color-adjust:exact;print-color-adjust:exact"></span>
        <span style="color:#DC2626;font-weight:700;font-size:7.5pt">ES_CONTRIBUCION</span>
      </span>
    </div>
    ${paras}
    <div class="footer">Generado por MAIA · Universidad de los Andes · ${new Date().toLocaleString('es-CO')}</div>
    <script>window.onload=function(){window.print();}<\/script>
    </body></html>`);
  win.document.close();
}

/* ── Init ── */
(async()=>{
  await Promise.all([loadModels(), loadLexicon()]);
  chkH(); setInterval(chkH,30000);
})();