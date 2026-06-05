/* ═══════════════════════════════════════════════
   IrisANN — Frontend Application Logic
   Handles: uploads, training API, predictions, chart
   ═══════════════════════════════════════════════ */

'use strict';

// ── State ──────────────────────────────────────
const state = {
  datasetLoaded: false,
  usingDefault:  false,
  modelTrained:  false,
  trainingHistory: null,
  classNames: ['setosa', 'versicolor', 'virginica'],
};

// ── Flower metadata ─────────────────────────────
const FLOWER_META = {
  setosa:     { emoji: '🌿', color: '#10b981', bar: 'linear-gradient(90deg,#10b981,#059669)' },
  versicolor: { emoji: '🌺', color: '#a855f7', bar: 'linear-gradient(90deg,#a855f7,#ec4899)' },
  virginica:  { emoji: '🌼', color: '#06b6d4', bar: 'linear-gradient(90deg,#06b6d4,#3b82f6)' },
};
const DEFAULT_BAR  = 'linear-gradient(90deg,#a855f7,#ec4899)';
const DEFAULT_COLOR = '#a855f7';

// ════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initDropZone('dz-dataset', 'fi-dataset', handleDatasetFile);
  initCardGlow();
  checkStatus();
});

// ════════════════════════════════════════════════
// CARD GLOW (mouse-follow radial gradient)
// ════════════════════════════════════════════════
function initCardGlow() {
  document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', ((e.clientX - r.left) / r.width  * 100) + '%');
      card.style.setProperty('--my', ((e.clientY - r.top)  / r.height * 100) + '%');
    });
  });
}

// ════════════════════════════════════════════════
// DRAG-AND-DROP FACTORY
// ════════════════════════════════════════════════
function initDropZone(zoneId, inputId, handler) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  if (!zone || !input) return;

  // Click on zone (not the inner link-btn)
  zone.addEventListener('click', e => {
    if (!e.target.classList.contains('link-btn')) input.click();
  });
  zone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') input.click();
  });

  // File input change
  input.addEventListener('change', () => {
    if (input.files[0]) handler(input.files[0]);
    input.value = '';
  });

  // Drag events
  ['dragenter','dragover'].forEach(ev => {
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('over'); });
  });
  ['dragleave','dragend'].forEach(ev => {
    zone.addEventListener(ev, () => zone.classList.remove('over'));
  });
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('over');
    const file = e.dataTransfer.files[0];
    if (file) handler(file);
  });
}

// ════════════════════════════════════════════════
// STEP TRAIL
// ════════════════════════════════════════════════
function setTrailStep(step) {
  [1, 2, 3].forEach(n => {
    const el = document.getElementById('trail-' + n);
    el.classList.remove('active', 'done');
    if (n < step)       el.classList.add('done');
    else if (n === step) el.classList.add('active');
  });
  [1, 2].forEach(n => {
    const line = document.getElementById(`line-${n}-${n+1}`);
    line.classList.toggle('done', n < step);
  });
}

// ════════════════════════════════════════════════
// GLOBAL STATUS BAR
// ════════════════════════════════════════════════
function setStatus(label, mode = 'idle') {
  const dot = document.querySelector('.gs-dot');
  const lbl = document.getElementById('gs-label');
  dot.className = 'gs-dot ' + mode;
  lbl.textContent = label;
}

// ════════════════════════════════════════════════
// CHECK SERVER STATUS
// ════════════════════════════════════════════════
async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) return;
    const d = await r.json();
    if (d.trained) {
      state.modelTrained  = true;
      state.classNames    = d.class_names;
      setStatus(`Đã huấn luyện · Accuracy ${d.accuracy}%`, 'trained');
      setTrailStep(3);
    }
  } catch (_) { /* server offline — OK */ }
}

// ════════════════════════════════════════════════
// DATASET — UPLOAD
// ════════════════════════════════════════════════
async function handleDatasetFile(file) {
  const allowed = ['csv','xlsx','xls'];
  const ext     = file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    return toast('❌ Chỉ hỗ trợ CSV và Excel', 'err');
  }

  toast('⏳ Đang đọc file…');
  const fd = new FormData();
  fd.append('file', file);

  try {
    const r = await fetch('/api/upload-dataset', { method: 'POST', body: fd });
    const d = await r.json();
    if (!d.success) return toast('❌ ' + d.error, 'err');

    state.datasetLoaded = true;
    state.usingDefault  = false;
    renderDatasetInfo(d.data);
    setTrailStep(2);
    toast('✅ Đã tải dataset thành công!', 'ok');
  } catch (e) {
    toast('❌ Lỗi kết nối server', 'err');
  }
}

function renderDatasetInfo(info) {
  $('ds-chip-name').textContent = '📄 ' + info.filename;
  $('ds-rows').textContent = info.rows + ' hàng';
  $('ds-cols').textContent = info.columns + ' cột';

  // Table head
  const thead = $('ds-thead');
  thead.innerHTML = '<tr>' + info.column_names.map(c => `<th>${c}</th>`).join('') + '</tr>';

  // Table body
  const tbody = $('ds-tbody');
  tbody.innerHTML = info.preview.map(row =>
    '<tr>' + info.column_names.map(c => `<td>${row[c] ?? '—'}</td>`).join('') + '</tr>'
  ).join('');

  show('ds-info');
}

// ════════════════════════════════════════════════
// DATASET — USE DEFAULT
// ════════════════════════════════════════════════
function useDefault() {
  state.datasetLoaded = true;
  state.usingDefault  = true;

  $('ds-chip-name').textContent = '🌿 Iris dataset (sklearn built-in)';
  $('ds-rows').textContent = '150 hàng';
  $('ds-cols').textContent  = '5 cột';

  $('ds-thead').innerHTML = '<tr><th>sepal length</th><th>sepal width</th><th>petal length</th><th>petal width</th><th>species</th></tr>';
  $('ds-tbody').innerHTML = `
    <tr><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
    <tr><td>6.0</td><td>2.9</td><td>4.5</td><td>1.5</td><td>versicolor</td></tr>
    <tr><td>6.7</td><td>3.1</td><td>5.6</td><td>2.4</td><td>virginica</td></tr>
    <tr><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
    <tr><td>5.8</td><td>2.7</td><td>5.1</td><td>1.9</td><td>virginica</td></tr>
  `;
  show('ds-info');
  setTrailStep(2);
  toast('✅ Đã chọn dataset Iris mặc định!', 'ok');
}

// ════════════════════════════════════════════════
// TRAIN MODEL
// ════════════════════════════════════════════════
async function trainModel() {
  if (!state.datasetLoaded) {
    return toast('⚠️ Vui lòng tải dataset trước!', 'err');
  }

  const btn    = $('btn-train');
  const inner  = $('btn-train-inner');
  const spin   = $('train-spin');

  btn.disabled = true;
  inner.textContent = 'Đang huấn luyện…';
  show('train-spin'); spin.classList.remove('hidden');
  setStatus('Đang huấn luyện…', 'busy');

  const body = {
    use_default: state.usingDefault,
    epochs:     parseInt($('hp-epochs').value) || 50,
    batch_size: parseInt($('hp-batch').value)  || 10,
    lr:         parseFloat($('hp-lr').value)   || 0.01,
  };

  try {
    const r = await fetch('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const d = await r.json();

    if (!d.success) {
      toast('❌ ' + d.error, 'err');
      setStatus('Lỗi huấn luyện', 'idle');
    } else {
      state.modelTrained  = true;
      state.classNames    = d.class_names;
      state.trainingHistory = d.history;

      $('kpi-acc').textContent  = d.accuracy + '%';
      $('kpi-loss').textContent = d.loss;
      $('kpi-tr').textContent   = d.train_size;
      $('kpi-ts').textContent   = d.test_size;

      show('train-results');
      drawChart(d.history);
      setTrailStep(3);
      setStatus(`Đã huấn luyện · Accuracy ${d.accuracy}%`, 'trained');
      toast(`✅ Huấn luyện xong! Accuracy: ${d.accuracy}%`, 'ok');
    }
  } catch (e) {
    toast('❌ Lỗi kết nối tới server Flask', 'err');
    setStatus('Lỗi kết nối', 'idle');
  } finally {
    btn.disabled = false;
    inner.textContent = '⚡ Huấn luyện lại';
    hide('train-spin');
  }
}

// ════════════════════════════════════════════════
// CHART — Canvas 2D (no library needed)
// ════════════════════════════════════════════════
function drawChart(history) {
  const canvas = $('chart');
  const ctx    = canvas.getContext('2d');
  const W = canvas.offsetWidth || 800;
  const H = 220;
  canvas.width  = W;
  canvas.height = H;

  const pad = { top: 20, right: 20, bottom: 36, left: 50 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;
  const n = history.loss.length;

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (innerH / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  function xPt(i) { return pad.left + (i / (n - 1)) * innerW; }
  function yPt(v, min, max) { return pad.top + (1 - (v - min) / (max - min || 1)) * innerH; }

  function drawLine(data, color, min = 0, max = 1) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';
    data.forEach((v, i) => {
      const x = xPt(i), y = yPt(v, min, max);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Glow
    ctx.shadowColor = color; ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Fill area (accuracy)
  const accMin = 0, accMax = 1;
  ctx.beginPath();
  history.accuracy.forEach((v, i) => {
    const x = xPt(i), y = yPt(v, accMin, accMax);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(xPt(n-1), pad.top + innerH);
  ctx.lineTo(xPt(0), pad.top + innerH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + innerH);
  grad.addColorStop(0, 'rgba(16,185,129,.25)');
  grad.addColorStop(1, 'rgba(16,185,129,0)');
  ctx.fillStyle = grad;
  ctx.fill();

  drawLine(history.accuracy, '#10b981', accMin, accMax);

  const lossMin = 0;
  const lossMax = Math.max(...history.loss) * 1.1;
  drawLine(history.loss, '#f87171', lossMin, lossMax);

  // X-axis labels
  ctx.fillStyle = 'rgba(148,163,184,.7)';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'center';
  const step = Math.ceil(n / 6);
  for (let i = 0; i < n; i += step) {
    ctx.fillText(i + 1, xPt(i), H - 8);
  }

  // Legend
  const legends = [
    { label: 'Accuracy', color: '#10b981' },
    { label: 'Loss',     color: '#f87171' },
  ];
  legends.forEach((l, i) => {
    const lx = pad.left + i * 110;
    ctx.beginPath();
    ctx.strokeStyle = l.color; ctx.lineWidth = 2.5;
    ctx.moveTo(lx, 10); ctx.lineTo(lx + 20, 10); ctx.stroke();
    ctx.fillStyle = l.color; ctx.textAlign = 'left';
    ctx.fillText(l.label, lx + 26, 14);
  });
}


// ════════════════════════════════════════════════
// QUICK FILL
// ════════════════════════════════════════════════
function fill(sl, sw, pl, pw) {
  $('f-sl').value = sl; $('f-sw').value = sw;
  $('f-pl').value = pl; $('f-pw').value = pw;
}

// ════════════════════════════════════════════════
// PREDICT — numeric input
// ════════════════════════════════════════════════
async function predictNums() {
  if (!state.modelTrained) return toast('⚠️ Vui lòng huấn luyện mô hình trước!', 'err');

  const features = [
    parseFloat($('f-sl').value),
    parseFloat($('f-sw').value),
    parseFloat($('f-pl').value),
    parseFloat($('f-pw').value),
  ];
  if (features.some(isNaN)) return toast('⚠️ Vui lòng nhập đầy đủ 4 đặc trưng', 'err');

  await runPrediction(features);
}


// ════════════════════════════════════════════════
// PREDICTION CORE
// ════════════════════════════════════════════════
async function runPrediction(features) {
  try {
    const r = await fetch('/api/predict-features', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    const d = await r.json();
    if (!d.success) return toast('❌ ' + d.error, 'err');

    renderPrediction(d);
  } catch (e) {
    toast('❌ Lỗi dự đoán', 'err');
  }
}

function renderPrediction(d) {
  const cls  = d.predicted_class.toLowerCase();
  const meta = FLOWER_META[cls] || { emoji: '🌸', color: DEFAULT_COLOR, bar: DEFAULT_BAR };

  // Image examples URLs from Wikimedia Commons
  const imageUrls = {
    'setosa': 'https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg',
    'versicolor': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg',
    'virginica': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
  };

  $('pred-emoji').textContent = meta.emoji;
  $('pred-name').textContent  = d.predicted_class.charAt(0).toUpperCase() + d.predicted_class.slice(1);
  $('pred-conf').textContent  = `Độ tự tin: ${d.confidence}%`;

  const imgSrc = imageUrls[cls];
  if (imgSrc) {
    $('pred-img-src').src = imgSrc;
    $('pred-img-src').style.display = 'block';
  } else {
    $('pred-img-src').style.display = 'none';
  }

  // Probability bars
  const list = $('prob-list');
  list.innerHTML = d.results.map(item => {
    const m   = FLOWER_META[item.class.toLowerCase()] || { bar: DEFAULT_BAR, color: DEFAULT_COLOR };
    const pct = item.probability;
    return `
      <div class="prob-item">
        <span class="prob-name">${item.class}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:0%;background:${m.bar}" data-pct="${pct}"></div>
        </div>
        <span class="prob-pct" style="color:${m.color}">${pct.toFixed(1)}%</span>
      </div>`;
  }).join('');

  show('pred-out');

  // Animate bars after render
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      list.querySelectorAll('.prob-fill').forEach(el => {
        el.style.width = el.dataset.pct + '%';
      });
    });
  });
}

// ════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ════════════════════════════════════════════════
let toastTimer;
function toast(msg, type = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className   = 'toast ' + type;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 3500);
}

// ════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════
const $    = id => document.getElementById(id);
const show = id => { const el = $(id); if (el) el.classList.remove('hidden'); };
const hide = id => { const el = $(id); if (el) el.classList.add('hidden'); };
