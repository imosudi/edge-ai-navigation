/**
 * dashboard/static/js/dashboard.js
 * Real-time dashboard client for Edge AI Navigation System.
 *
 * Manages four WebSocket channels:
 *   /api/v1/ws/camera    - JPEG binary frames → <img> update via Blob URL
 *   /api/v1/ws/lidar     - JSON scan data     → polar canvas render
 *   /api/v1/ws/fusion    - JSON fused objects → objects table update
 *   /api/v1/ws/telemetry - JSON system metrics → gauge + metric update
 *
 * All WebSockets reconnect automatically on close/error with
 * exponential back-off (1s → 2s → 4s … max 30s).
 */

'use strict';

/* ─── Constants ──────────────────────────────────────────────── */
const WS_BASE       = `ws://${location.host}/api/v1`;
const MAX_LOG_LINES = 200;
const GAUGE_SIZE    = 64;   // canvas px

/* ─── State ──────────────────────────────────────────────────── */
let lidarCtx        = null;
let prevBlobUrl     = null;
let logLines        = 0;
let lastLidarFpsTs  = 0;
let lidarFrameCount = 0;
let lastLidarData   = null;
let lastTelemetryData = null;
let lastObjects     = [];
const uniqueClasses = new Set();

/* ─── DOM refs ───────────────────────────────────────────────── */
const $feed     = document.getElementById('camera-feed');
const $log      = document.getElementById('event-log');
const $tbody    = document.getElementById('objects-tbody');
const $connDot  = document.getElementById('conn-dot');
const $connLbl  = document.getElementById('conn-label');

/* ─── Clock ──────────────────────────────────────────────────── */
function tickClock() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-GB', { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

/* ─── Logging ────────────────────────────────────────────────── */
function addLog(msg, cls = 'log-info') {
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const el = document.createElement('div');
  el.className = 'log-entry';
  el.innerHTML = `<span class="log-ts">${ts}</span><span class="${cls}">${escHtml(msg)}</span>`;
  $log.prepend(el);
  if (++logLines > MAX_LOG_LINES) {
    $log.removeChild($log.lastChild);
    logLines--;
  }
}
function clearLog() { $log.innerHTML = ''; logLines = 0; }
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ─── WebSocket factory with auto-reconnect ──────────────────── */
function makeWS(path, onMsg, onBinMsg, label) {
  let ws, delay = 1000;

  function connect() {
    ws = new WebSocket(`${WS_BASE}${path}`);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      delay = 1000;
      addLog(`Connected: ${label}`, 'log-info');
      setConnStatus(true);
    };
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        onBinMsg && onBinMsg(ev.data);
      } else {
        try { onMsg && onMsg(JSON.parse(ev.data)); }
        catch(e) { /* ignore bad JSON */ }
      }
    };
    ws.onclose = () => {
      addLog(`Disconnected: ${label} - retry in ${delay/1000}s`, 'log-warn');
      setConnStatus(false);
      setTimeout(connect, delay);
      delay = Math.min(delay * 2, 30000);
    };
    ws.onerror = () => ws.close();
    // Send periodic ping to keep connection alive
    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
      else clearInterval(ping);
    }, 20000);
  }
  connect();
}

/* ─── Connection status indicator ────────────────────────────── */
let connectedChannels = 0;
function setConnStatus(up) {
  connectedChannels = Math.max(0, connectedChannels + (up ? 1 : -1));
  if (connectedChannels > 0) {
    $connDot.className = 'status-dot ok';
    $connLbl.textContent = `${connectedChannels} channel(s) live`;
  } else {
    $connDot.className = 'status-dot error';
    $connLbl.textContent = 'Disconnected';
  }
}

/* ─── Camera WebSocket ───────────────────────────────────────── */
/* ─── Camera & LiDAR WebSockets (moved to initWebSockets) ─── */
const lidarCanvas = document.getElementById('lidar-canvas');
lidarCtx = lidarCanvas.getContext('2d');

function renderLidar(data) {
  const W = lidarCanvas.width, H = lidarCanvas.height;
  const cx = W / 2, cy = H / 2;
  const maxR = Math.min(cx, cy) - 10;
  const maxDist = 5.5; // m
  const scale = maxR / maxDist;

  const ctx = lidarCtx;
  ctx.clearRect(0, 0, W, H);

  const isLight = !document.body.classList.contains('dark-theme');

  // Background
  ctx.fillStyle = isLight ? '#eaedf2' : '#0a0c10';
  ctx.fillRect(0, 0, W, H);

  // Grid rings
  ctx.strokeStyle = isLight ? '#ccd1db' : '#1e2535';
  ctx.lineWidth = 1;
  [1, 2, 3, 4, 5].forEach(r => {
    ctx.beginPath();
    ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = isLight ? '#57606a' : '#2a3450';
    ctx.font = '9px monospace';
    ctx.fillText(`${r}m`, cx + r * scale + 2, cy);
  });

  // Crosshair
  ctx.strokeStyle = isLight ? '#ccd1db' : '#1e2535';
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();

  // FOV arc lines (−120° to +120°)
  ctx.strokeStyle = isLight ? '#9eb0c7' : '#2a3450';
  ctx.lineWidth = 1;
  [-120, 120].forEach(deg => {
    const rad = (deg - 90) * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(rad) * maxR, cy + Math.sin(rad) * maxR);
    ctx.stroke();
  });

  // Obstacle zone highlights
  const zones = data.obstacle_zones || [];
  zones.forEach(z => {
    const rad = (z.angle_deg - 90) * Math.PI / 180;
    const r   = Math.min(z.distance_m, maxDist) * scale;
    const col = z.threat_level === 'HIGH'   ? 'rgba(255,23,68,0.15)' :
                z.threat_level === 'MEDIUM' ? 'rgba(255,109,0,0.10)' :
                                              'rgba(0,230,118,0.06)';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r + 12, rad - 0.15, rad + 0.15);
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();
  });

  // LiDAR points
  const angles = data.angles_deg || [];
  const dists  = data.distances_m || [];

  for (let i = 0; i < angles.length; i++) {
    const angleDeg = angles[i];
    const distM    = dists[i];
    const r = Math.min(distM, maxDist) * scale;
    const rad = (angleDeg - 90) * Math.PI / 180;
    const px = cx + r * Math.cos(rad);
    const py = cy + r * Math.sin(rad);

    // Colour by distance
    const t = 1 - distM / maxDist;
    const rr = Math.round(255 * t);
    const gg = Math.round(200 * (1 - t));
    ctx.fillStyle = `rgb(${rr},${255 - rr},${gg})`;
    ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
  }

  // Robot origin
  ctx.fillStyle = '#00e5ff';
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, Math.PI * 2);
  ctx.fill();

  // Scan count watermark
  ctx.fillStyle = isLight ? '#57606a' : '#2a3450';
  ctx.font = '10px monospace';
  ctx.fillText(`scan #${data.scan_count || 0}`, 6, H - 6);
}

/* ─── Fusion WebSocket (moved to initWebSockets) ─── */
let infFpsCount = 0, lastInfFpsTs = performance.now();

function renderObjects(objects) {
  const filterSelect = document.getElementById('class-filter');
  const filterVal = filterSelect ? filterSelect.value : 'all';

  const filtered = filterVal === 'all'
    ? objects
    : objects.filter(o => o.class_name === filterVal);

  const rows = filtered.map(o => {
    const dist  = o.distance_m != null ? o.distance_m.toFixed(2) + ' m' : '--';
    const conf  = (o.confidence * 100).toFixed(0) + '%';
    const tcls  = `threat-${o.threat_level}`;
    return `<tr>
      <td>#${o.track_id}</td>
      <td>${escHtml(o.class_name)}</td>
      <td>${conf}</td>
      <td>${dist}</td>
      <td>${escHtml(o.direction)}</td>
      <td class="${tcls}">${escHtml(o.threat_level)}</td>
    </tr>`;
  }).join('');
  $tbody.innerHTML = rows;

  const totalCount = objects.length;
  const filteredCount = filtered.length;
  const countBadge = document.getElementById('obj-count');
  if (countBadge) {
    if (filterVal === 'all') {
      countBadge.textContent = `${totalCount} object${totalCount !== 1 ? 's' : ''}`;
    } else {
      countBadge.textContent = `${filteredCount}/${totalCount} object${totalCount !== 1 ? 's' : ''}`;
    }
  }
}

function updateClassFilter(objects) {
  let changed = false;
  objects.forEach(o => {
    if (o.class_name && !uniqueClasses.has(o.class_name)) {
      uniqueClasses.add(o.class_name);
      changed = true;
    }
  });

  if (changed) {
    const filterSelect = document.getElementById('class-filter');
    if (filterSelect) {
      const selectedValue = filterSelect.value;
      filterSelect.innerHTML = '<option value="all">All Classes</option>';
      const sortedClasses = Array.from(uniqueClasses).sort();
      sortedClasses.forEach(cls => {
        const opt = document.createElement('option');
        opt.value = cls;
        opt.textContent = cls;
        filterSelect.appendChild(opt);
      });
      if (uniqueClasses.has(selectedValue)) {
        filterSelect.value = selectedValue;
      } else {
        filterSelect.value = 'all';
      }
    }
  }
}

function updateNavigation(nav) {
  const $navAction = document.getElementById('nav-action');
  const $navSpeed = document.getElementById('nav-speed');
  const $navReason = document.getElementById('nav-reason');

  if (!$navAction || !$navSpeed || !$navReason) return;

  const action = nav?.action || 'MOVE_FORWARD';
  const speed = nav?.speed != null ? nav.speed : 1.0;
  const reason = nav?.reason || 'Path clear';

  $navAction.textContent = action;
  $navSpeed.textContent = speed.toFixed(2);
  $navReason.textContent = reason;

  let color = 'var(--text-dim)';
  if (action === 'MOVE_FORWARD' || action === 'FORWARD') {
    color = 'var(--green)';
  } else if (action === 'STEER_LEFT' || action === 'STEER_RIGHT') {
    color = 'var(--orange)';
  } else if (action === 'STOP') {
    color = 'var(--red)';
  }

  $navAction.style.color = color;
  $navAction.style.borderColor = `color-mix(in srgb, ${color} 35%, transparent)`;
  $navAction.style.backgroundColor = `color-mix(in srgb, ${color} 8%, transparent)`;
}

function updateTelemetry(data) {
  // Gauge arcs
  updateGauge('g-cpu',  data.cpu?.percent ?? 0,          100, '#00e5ff');
  updateGauge('g-mem',  data.memory?.percent ?? 0,        100, '#7c4dff');
  updateGauge('g-temp', data.temperature?.cpu_c ?? 0,      85, temp => temp > 75 ? '#ff1744' : temp > 60 ? '#ff6d00' : '#00e676');
  updateGauge('g-disk', data.disk?.percent ?? 0,          100, '#00e676');

  // Gauge numeric values
  document.getElementById('val-cpu').textContent  = (data.cpu?.percent ?? '--') + '%';
  document.getElementById('val-mem').textContent  = (data.memory?.percent ?? '--') + '%';
  document.getElementById('val-temp').textContent = (data.temperature?.cpu_c ?? '--') + '°C';
  document.getElementById('val-disk').textContent = (data.disk?.percent ?? '--') + '%';

  // Metric rows
  const uptime = data.uptime_seconds;
  document.getElementById('val-uptime').textContent = uptime != null
    ? formatUptime(uptime) : '--';

  document.getElementById('val-fps-cam').textContent =
    (data.fps?.camera ?? '--') + ' fps';
  document.getElementById('fps-inference').textContent =
    (data.fps?.inference ?? '--') + ' fps';
  document.getElementById('val-fps-fusion').textContent =
    (data.fps?.fusion ?? '--') + ' fps';

  document.getElementById('val-net-recv').textContent =
    data.network?.recv_kbps != null ? data.network.recv_kbps.toFixed(1) + ' KB/s' : '--';
  document.getElementById('val-net-sent').textContent =
    data.network?.sent_kbps != null ? data.network.sent_kbps.toFixed(1) + ' KB/s' : '--';

  // Hailo / Engine stats
  const h = data.hailo;
  const engine = h?.device_type || 'cpu';

  const engineLabel = {
    'cpu': 'CPU Fallback',
    'gpu': 'GPU Acceleration',
    'npu': 'NPU (Hailo-8L)'
  }[engine] || 'CPU Fallback';

  // Update header badge
  const $engineBadge = document.getElementById('val-engine-type');
  if ($engineBadge) {
    $engineBadge.textContent = engineLabel;
    if (engine === 'npu') {
      $engineBadge.style.color = '#00e5ff';
      $engineBadge.style.background = 'rgba(0,229,255,0.08)';
    } else if (engine === 'gpu') {
      $engineBadge.style.color = '#ffb300';
      $engineBadge.style.background = 'rgba(255,179,0,0.08)';
    } else {
      $engineBadge.style.color = '#7c4dff';
      $engineBadge.style.background = 'rgba(124,77,255,0.08)';
    }
  }

  // Update telemetry row
  document.getElementById('val-hailo').textContent = engineLabel;
  if (h?.available || engine === 'npu') {
    document.getElementById('val-inf-lat').textContent =
      (h.last_latency_ms ?? '--') + ' ms';
  } else {
    document.getElementById('val-inf-lat').textContent = '--';
  }
}

/* ─── Telemetry WebSocket (moved to initWebSockets) ─── */

/* ─── Gauge arc renderer ─────────────────────────────────────── */
const _gaugeCtxCache = {};

function updateGauge(wrapperId, value, max, colourOrFn) {
  const wrapper = document.getElementById(wrapperId);
  if (!wrapper) return;
  const canvas = wrapper.querySelector('canvas.gauge-arc');
  if (!canvas) return;

  let ctx = _gaugeCtxCache[wrapperId];
  if (!ctx) {
    canvas.width  = GAUGE_SIZE;
    canvas.height = GAUGE_SIZE;
    ctx = canvas.getContext('2d');
    _gaugeCtxCache[wrapperId] = ctx;
  }

  const cx = GAUGE_SIZE / 2, cy = GAUGE_SIZE / 2;
  const r  = cx - 5;
  const startAngle = Math.PI * 0.75;
  const fullAngle  = Math.PI * 1.5;
  const fraction   = Math.min(Math.max(value / max, 0), 1);

  const colour = typeof colourOrFn === 'function' ? colourOrFn(value) : colourOrFn;

  ctx.clearRect(0, 0, GAUGE_SIZE, GAUGE_SIZE);

  const isLight = !document.body.classList.contains('dark-theme');

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + fullAngle);
  ctx.strokeStyle = isLight ? '#d0d7de' : '#1e2535';
  ctx.lineWidth   = 6;
  ctx.lineCap     = 'round';
  ctx.stroke();

  // Fill
  if (fraction > 0) {
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, startAngle + fullAngle * fraction);
    ctx.strokeStyle = colour;
    ctx.lineWidth   = 6;
    ctx.lineCap     = 'round';
    ctx.stroke();
  }
}

/* ─── Helpers ─────────────────────────────────────────────────── */
function formatUptime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

/* ─── Initial log entry ──────────────────────────────────────── */
addLog('Dashboard initialised - connecting to Edge AI Navigation System…', 'log-info');

/* ─── Theme toggle ───────────────────────────────────────────── */
const SUN_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: block;"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>`;

const MOON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: block;"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`;

const $themeToggle = document.getElementById('theme-toggle');
if ($themeToggle) {
  // Check persisted preference or system preference
  const savedTheme = localStorage.getItem('theme');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const useDark = savedTheme === 'dark' || (!savedTheme && systemPrefersDark);

  if (useDark) {
    document.body.classList.add('dark-theme');
    $themeToggle.innerHTML = SUN_SVG;
  } else {
    document.body.classList.remove('dark-theme');
    $themeToggle.innerHTML = MOON_SVG;
  }

  $themeToggle.addEventListener('click', () => {
    const isDark = document.body.classList.toggle('dark-theme');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    $themeToggle.innerHTML = isDark ? SUN_SVG : MOON_SVG;

    // Redraw LiDAR immediately if data is cached
    if (lastLidarData) {
      renderLidar(lastLidarData);
    }
    // Redraw telemetry gauges immediately if telemetry is cached
    if (lastTelemetryData) {
      updateTelemetry(lastTelemetryData);
    }
  });
}

// Bind change listener for dynamic class filtering
const $classFilter = document.getElementById('class-filter');
if ($classFilter) {
  $classFilter.addEventListener('change', () => {
    renderObjects(lastObjects);
  });
}

// Fetch status on startup and initialize WebSockets
let deviceStatus = { camera: true, lidar: true };

function initWebSockets() {
  /* ─── Camera WebSocket ───────────────────────────────────────── */
  makeWS('/ws/camera', null, (buf) => {
    const blob = new Blob([buf], { type: 'image/jpeg' });
    const url  = URL.createObjectURL(blob);
    $feed.src  = url;
    if (prevBlobUrl) URL.revokeObjectURL(prevBlobUrl);
    prevBlobUrl = url;
  }, 'Camera');

  /* ─── LiDAR WebSocket ────────────────────────────────────────── */
  if (deviceStatus.lidar) {
    makeWS('/ws/lidar', (data) => {
      lastLidarData = data;
      renderLidar(data);
      const count = (data.angles_deg || []).length;
      document.getElementById('lidar-points').textContent = `pts: ${count}`;
      const minD = data.min_distance_m != null ? data.min_distance_m.toFixed(2) + ' m' : '--';
      document.getElementById('lidar-min-dist').textContent = `min: ${minD}`;

      // Local FPS calculation
      const now = performance.now();
      lidarFrameCount++;
      if (now - lastLidarFpsTs >= 1000) {
        document.getElementById('fps-lidar').textContent =
          `${lidarFrameCount} fps`;
        lidarFrameCount = 0;
        lastLidarFpsTs = now;
      }
    }, null, 'LiDAR');
  } else {
    addLog('LiDAR hardware disconnected - stream suspended', 'log-warn');
    document.getElementById('fps-lidar').textContent = 'offline';
    document.getElementById('lidar-min-dist').textContent = 'min: --';
    document.getElementById('lidar-points').textContent = 'pts: --';
  }

  /* ─── Fusion WebSocket ───────────────────────────────────────── */
  makeWS('/ws/fusion', (data) => {
    const objects = data.objects || [];
    lastObjects = objects;
    updateClassFilter(objects);
    renderObjects(objects);
    updateNavigation(data.navigation);

    // Log new HIGH threats
    objects.filter(o => o.threat_level === 'HIGH').forEach(o => {
      addLog(
        `⚠ HIGH THREAT: ${o.class_name} at ${(o.distance_m || 0).toFixed(2)}m (${o.direction})`,
        'log-threat-HIGH'
      );
    });
  }, null, 'Fusion');

  /* ─── Telemetry WebSocket ────────────────────────────────────── */
  makeWS('/ws/telemetry', (data) => {
    lastTelemetryData = data;
    updateTelemetry(data);
  }, null, 'Telemetry');
}

fetch('/api/v1/status')
  .then(res => res.json())
  .then(data => {
    if (data.components) {
      deviceStatus.camera = data.components.camera === 'ok';
      deviceStatus.lidar = data.components.lidar === 'ok';
    }
    initWebSockets();
  })
  .catch(err => {
    console.warn('Could not fetch status, defaulting to all available:', err);
    initWebSockets();
  });
