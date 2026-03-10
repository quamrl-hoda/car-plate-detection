/* PlateVision — main.js */

const dz         = document.getElementById('dz');
const dzIdle     = document.getElementById('dzIdle');
const dzTypes    = document.getElementById('dzTypes');
const fileInput  = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const previewVid = document.getElementById('previewVid');
const fileMeta   = document.getElementById('fileMeta');
const fmName     = document.getElementById('fmName');
const fmSize     = document.getElementById('fmSize');
const fmType     = document.getElementById('fmType');
const runBtn     = document.getElementById('runBtn');
const runText    = document.getElementById('runText');
const runDots    = document.getElementById('runDots');

const rIdle      = document.getElementById('rIdle');
const rImage     = document.getElementById('rImage');
const rVideo     = document.getElementById('rVideo');
const rError     = document.getElementById('rError');
const chip       = document.getElementById('chip');
const outImg     = document.getElementById('outImg');
const outVid     = document.getElementById('outVid');
const dlBtn      = document.getElementById('dlBtn');
const detN       = document.getElementById('detN');
const detList    = document.getElementById('detList');
const latency    = document.getElementById('latency');
const errMsg     = document.getElementById('errMsg');
const iconImg    = document.getElementById('iconImg');
const iconVid    = document.getElementById('iconVid');
const dot        = document.getElementById('dot');
const dotLabel   = document.getElementById('dotLabel');
const clock      = document.getElementById('clock');
const ocrSection = document.getElementById('ocrSection');
const ocrPlates  = document.getElementById('ocrPlates');

let mode = 'image', currentFile = null;

setInterval(() => { clock.textContent = new Date().toLocaleTimeString(); }, 1000);

// ── Health ────────────────────────────────────────────────────────
(async function checkHealth() {
  try {
    const d = await fetch('/health').then(r => r.json());
    dot.className        = d.model_ready ? 'dot ready' : 'dot error';
    dotLabel.textContent = d.model_ready ? 'Model ready' : 'Model not loaded';
  } catch {
    dot.className = 'dot error';
    dotLabel.textContent = 'Server offline';
  }
})();

// ── Mode ──────────────────────────────────────────────────────────
function setMode(m) {
  mode = m;
  document.getElementById('btnImage').classList.toggle('active', m==='image');
  document.getElementById('btnVideo').classList.toggle('active', m==='video');
  iconImg.classList.toggle('hidden', m!=='image');
  iconVid.classList.toggle('hidden', m!=='video');
  dzTypes.textContent = m==='image' ? 'JPG · PNG · WEBP · BMP' : 'MP4 · AVI · MOV · MKV';
  fileInput.accept    = m==='image' ? 'image/*' : 'video/*';
  runText.textContent = 'Run Inference';
  resetResult(); clearFile();
}

// ── File ─────────────────────────────────────────────────────────
dz.addEventListener('click',    () => fileInput.click());
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
dz.addEventListener('dragleave',    () => dz.classList.remove('over'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('over'); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function fmt(b) { return b < 1048576 ? (b/1024).toFixed(1)+' KB' : (b/1048576).toFixed(1)+' MB'; }

function loadFile(file) {
  currentFile = file;
  const url = URL.createObjectURL(file);
  dzIdle.classList.add('hidden');
  previewImg.classList.add('hidden');
  previewVid.classList.add('hidden');
  if (file.type.startsWith('video/')) { previewVid.src=url; previewVid.classList.remove('hidden'); }
  else { previewImg.src=url; previewImg.classList.remove('hidden'); }
  fmName.textContent = file.name.length>24 ? file.name.slice(0,22)+'…' : file.name;
  fmSize.textContent = fmt(file.size);
  fmType.textContent = file.type || file.name.split('.').pop().toUpperCase();
  fileMeta.classList.remove('hidden');
  runBtn.disabled = false;
  resetResult();
}

function clearFile() {
  currentFile=null; previewImg.classList.add('hidden'); previewVid.classList.add('hidden');
  dzIdle.classList.remove('hidden'); fileMeta.classList.add('hidden'); runBtn.disabled=true;
}

function resetResult() {
  rIdle.classList.remove('hidden');
  rImage.classList.add('hidden'); rVideo.classList.add('hidden'); rError.classList.add('hidden');
  chip.className='result-chip'; chip.textContent='—';
}

// ── Run ───────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  setLoading(true);
  try { currentFile.type.startsWith('video/') ? await runVideo() : await runImage(); }
  catch(e) { showError(e.message || 'Unexpected error'); }
  finally { setLoading(false); }
});

function setLoading(on) {
  runBtn.disabled = on;
  runText.classList.toggle('hidden', on);
  runDots.classList.toggle('hidden', !on);
}

// ── Image inference ───────────────────────────────────────────────
async function runImage() {
  const fd = new FormData();
  fd.append('file', currentFile);
  const res  = await fetch('/predict', { method:'POST', body:fd });
  const data = await res.json();
  if (!res.ok) { showError(data.error || 'Inference failed'); return; }

  // Show annotated image
  outImg.src = 'data:image/jpeg;base64,' + data.image_b64;
  rIdle.classList.add('hidden');
  rImage.classList.remove('hidden');

  // Detection count
  detN.textContent = data.total;
  latency.textContent = data.latency_ms + 'ms';

  // Badge
  chip.textContent = data.total > 0 ? `${data.total} plate${data.total>1?'s':''} found` : 'No plates';
  chip.className   = 'result-chip ' + (data.total>0 ? 'ok' : 'none');

  // ── OCR results — prominent display ──────────────────────────────
  ocrPlates.innerHTML = '';
  const platesWithText = data.detections.filter(d => d.plate_text && d.plate_text.trim());

  if (platesWithText.length > 0) {
    ocrSection.classList.remove('hidden');
    platesWithText.forEach(d => {
      const card = document.createElement('div');
      card.className = 'ocr-plate-card';
      card.innerHTML = `
        <span class="ocr-plate-text">${d.plate_text}</span>
        <span class="ocr-plate-conf">${Math.round(d.confidence*100)}% confidence</span>`;
      ocrPlates.appendChild(card);
    });
  } else if (data.total > 0) {
    // Plate detected but OCR couldn't read it
    ocrSection.classList.remove('hidden');
    ocrPlates.innerHTML = '<span class="ocr-plate-none">Plate detected but text unreadable — install tesseract-ocr for OCR</span>';
  } else {
    ocrSection.classList.add('hidden');
  }

  // Detection confidence rows
  detList.innerHTML = '';
  data.detections.forEach((d, i) => {
    const pct  = Math.round(d.confidence * 100);
    const bbox = d.bbox.map(v => Math.round(v)).join(', ');
    const row  = document.createElement('div');
    row.className = 'det-row';
    row.innerHTML = `
      <span class="det-class">#${i+1} ${d.class_name}</span>
      <div class="det-bar-w"><div class="det-bar" style="width:${pct}%"></div></div>
      <span class="det-conf">${pct}%</span>
      <span class="det-bbox">[${bbox}]</span>`;
    detList.appendChild(row);
  });
}

// ── Video inference ───────────────────────────────────────────────
async function runVideo() {
  const fd = new FormData();
  fd.append('file', currentFile);
  const res = await fetch('/predict/video', { method:'POST', body:fd });
  if (!res.ok) { const d = await res.json().catch(()=>({})); showError(d.error||'Video failed'); return; }
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  outVid.src = url; dlBtn.href = url;
  dlBtn.classList.remove('hidden');
  rIdle.classList.add('hidden'); rVideo.classList.remove('hidden');
  chip.textContent='Video processed'; chip.className='result-chip ok';
}

// ── Error ─────────────────────────────────────────────────────────
function showError(msg) {
  rIdle.classList.add('hidden'); rImage.classList.add('hidden'); rVideo.classList.add('hidden');
  rError.classList.remove('hidden');
  errMsg.textContent=msg; chip.textContent='Error'; chip.className='result-chip none';
}
