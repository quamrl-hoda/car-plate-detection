/* PlateVision — main.js */

// ── DOM refs ─────────────────────────────────────────────────────
const dropzone     = document.getElementById('dropzone');
const dropIdle     = document.getElementById('dropIdle');
const fileInput    = document.getElementById('fileInput');
const previewImg   = document.getElementById('previewImg');
const fileMeta     = document.getElementById('fileMeta');
const metaName     = document.getElementById('metaName');
const metaSize     = document.getElementById('metaSize');
const metaDims     = document.getElementById('metaDims');
const runBtn       = document.getElementById('runBtn');
const btnText      = runBtn.querySelector('.btn-text');
const btnLoader    = document.getElementById('btnLoader');

const resultIdle    = document.getElementById('resultIdle');
const resultContent = document.getElementById('resultContent');
const resultBadge   = document.getElementById('resultBadge');
const annotatedImg  = document.getElementById('annotatedImg');
const detCount      = document.getElementById('detCount');
const detectionsList = document.getElementById('detectionsList');
const latencyVal    = document.getElementById('latencyVal');
const rawJson       = document.getElementById('rawJson');

const statusDot   = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');

const retrainBtn  = document.getElementById('retrainBtn');
const trainLog    = document.getElementById('trainLog');
const logContent  = document.getElementById('logContent');

const footerClock = document.getElementById('footerClock');

// ── State ─────────────────────────────────────────────────────────
let currentFile = null;

// ── Clock ─────────────────────────────────────────────────────────
function updateClock() {
  footerClock.textContent = new Date().toLocaleTimeString();
}
updateClock();
setInterval(updateClock, 1000);

// ── Health check ──────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    if (data.model_ready) {
      statusDot.className   = 'status-dot ready';
      statusLabel.textContent = 'Model ready';
    } else {
      statusDot.className   = 'status-dot error';
      statusLabel.textContent = 'Model not loaded';
    }
  } catch {
    statusDot.className   = 'status-dot error';
    statusLabel.textContent = 'Server offline';
  }
}
checkHealth();

// ── Dropzone ──────────────────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function loadFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    previewImg.src = dataUrl;
    previewImg.classList.remove('hidden');
    dropIdle.classList.add('hidden');

    // Get image dimensions
    const img = new Image();
    img.onload = () => {
      metaDims.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
    };
    img.src = dataUrl;

    metaName.textContent = file.name;
    metaSize.textContent = formatBytes(file.size);
    fileMeta.classList.remove('hidden');
    runBtn.disabled = false;

    // Reset result panel
    resetResults();
  };
  reader.readAsDataURL(file);
}

// ── Reset result panel ────────────────────────────────────────────
function resetResults() {
  resultIdle.classList.remove('hidden');
  resultContent.classList.add('hidden');
  resultBadge.className = 'result-badge';
  resultBadge.textContent = '—';
}

// ── Run Inference ─────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  // Loading state
  btnText.classList.add('hidden');
  btnLoader.classList.remove('hidden');
  runBtn.disabled = true;

  try {
    const reader = new FileReader();
    reader.onload = async e => {
      const base64 = e.target.result; // includes data:image/...;base64,

      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 }),
      });

      const data = await res.json();

      if (!res.ok) {
        alert('Inference error: ' + (data.error || 'Unknown error'));
        return;
      }

      renderResults(data);
    };
    reader.readAsDataURL(currentFile);
  } finally {
    btnText.classList.remove('hidden');
    btnLoader.classList.add('hidden');
    runBtn.disabled = false;
  }
});

// ── Render results ────────────────────────────────────────────────
function renderResults(data) {
  const { image_b64, detections, total, latency_ms } = data;

  // Show annotated image
  annotatedImg.src = 'data:image/jpeg;base64,' + image_b64;

  // Detection count
  detCount.textContent = total;

  // Badge
  resultBadge.textContent = total > 0
    ? `${total} plate${total > 1 ? 's' : ''} found`
    : 'No plates found';
  resultBadge.className = 'result-badge ' + (total > 0 ? 'detected' : 'none');

  // Detection list
  detectionsList.innerHTML = '';
  detections.forEach(det => {
    const pct  = Math.round(det.confidence * 100);
    const bbox = det.bbox.map(v => Math.round(v)).join(', ');
    const row  = document.createElement('div');
    row.className = 'det-row';
    row.innerHTML = `
      <span class="det-class">${det.class_name}</span>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:${pct}%"></div></div>
      <span class="det-conf">${pct}%</span>
    `;
    detectionsList.appendChild(row);
  });

  // Latency
  latencyVal.textContent = latency_ms + 'ms';

  // Raw JSON — hide image_b64 to keep it readable
  const display = { ...data };
  display.image_b64 = '[base64 omitted]';
  rawJson.textContent = JSON.stringify(display, null, 2);

  // Show result panel
  resultIdle.classList.add('hidden');
  resultContent.classList.remove('hidden');
}

// ── Retrain ───────────────────────────────────────────────────────
retrainBtn.addEventListener('click', async () => {
  retrainBtn.disabled = true;
  retrainBtn.textContent = '⏳ Training…';
  trainLog.classList.remove('hidden');
  logContent.textContent = 'Starting pipeline…\n';

  try {
    const res  = await fetch('/train');
    const data = await res.json();

    if (data.status === 'success') {
      logContent.textContent = data.stdout || 'Training complete.';
      retrainBtn.textContent = '✓ Done — click to retrain';
      checkHealth(); // refresh model status
    } else {
      logContent.textContent = data.stderr || data.error || 'Training failed.';
      retrainBtn.textContent = '✗ Failed — click to retry';
    }
  } catch (err) {
    logContent.textContent = 'Network error: ' + err.message;
    retrainBtn.textContent = '✗ Failed — click to retry';
  } finally {
    retrainBtn.disabled = false;
  }
});