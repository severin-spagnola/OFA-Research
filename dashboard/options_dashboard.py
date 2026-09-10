"""OverfitAlpha options trading dashboard — quant theme (white/grey/magenta).

Single Python file that generates a self-contained HTML/CSS/JS dashboard.
Mirrors the architecture of infra/alpha_dashboard.py from the RTH repo.
"""

import json


def options_dashboard_html() -> str:
    return _CSS + _BODY + _JS + "</body></html>"


# ── CSS ──────────────────────────────────────────────────────────────────────

_CSS = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OverfitAlpha — Options</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'SF Mono','Fira Code','Cascadia Code','Consolas',monospace;background:#f5f5f7;color:#1d1d1f;padding:0}
a{color:#d6336c;text-decoration:none}

/* Header */
.hdr{background:#1d1d1f;border-bottom:2px solid #d6336c;padding:14px 24px;display:flex;align-items:center;justify-content:space-between}
.hdr h1{color:#fff;font-size:1.15em;letter-spacing:.08em;text-transform:uppercase}
.hdr h1 span{color:#d6336c;font-weight:400;font-size:.65em;margin-left:10px;text-transform:none;letter-spacing:0}
.hdr-right{display:flex;gap:14px;align-items:center;font-size:.75em;color:#8e8e93}
.hdr-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px}
.dot-live{background:#34c759}.dot-paper{background:#ff9500}.dot-off{background:#8e8e93}

/* Status indicator */
.status-pill{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:12px;font-size:.7em;font-weight:600}
.pill-active{background:#34c75920;color:#34c759}
.pill-sim{background:#007aff20;color:#007aff}

/* Tabs */
.tabs{display:flex;gap:0;background:#fff;border-bottom:1px solid #e5e5ea;padding:0 24px;overflow-x:auto}
.tab{padding:10px 20px;cursor:pointer;color:#8e8e93;font-size:.8em;border-bottom:2px solid transparent;transition:all .15s;white-space:nowrap;text-transform:uppercase;letter-spacing:.06em}
.tab:hover{color:#1d1d1f}
.tab.active{color:#d6336c;border-bottom-color:#d6336c;font-weight:600}

/* Layout */
.main{padding:20px 24px;max-width:1480px;margin:0 auto}
.section{margin-bottom:20px}
.section-title{font-size:.75em;text-transform:uppercase;letter-spacing:.1em;color:#8e8e93;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #e5e5ea}

/* Cards */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:16px}
.grid-wide{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.card{background:#fff;border:1px solid #e5e5ea;border-radius:8px;padding:14px}
.card h3{color:#8e8e93;font-size:.7em;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.card .val{font-size:1.5em;font-weight:700}

/* Colors */
.mg{color:#d6336c}.green{color:#34c759}.red{color:#ff3b30}.blue{color:#007aff}.yellow{color:#ff9500}.gray{color:#8e8e93}
.bg-mg{background:#d6336c;color:#fff}
.pnl{font-weight:700}

/* Metric cards */
.m-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.m-card{background:#fff;border:1px solid #e5e5ea;border-radius:8px;padding:12px}
.m-card .m-label{font-size:.65em;color:#8e8e93;text-transform:uppercase;letter-spacing:.06em}
.m-card .m-val{font-size:1.3em;font-weight:700;margin-top:3px}

/* Tables */
table{width:100%;border-collapse:collapse;font-size:.8em}
th{text-align:left;padding:8px 10px;background:#f9f9fb;color:#8e8e93;font-size:.7em;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid #e5e5ea;position:sticky;top:0}
td{padding:7px 10px;border-bottom:1px solid #f2f2f7}
tr:hover td{background:#f9f9fb}

/* Badges */
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.7em;font-weight:600}
.badge-win{background:#34c75920;color:#34c759}
.badge-loss{background:#ff3b3020;color:#ff3b30}
.badge-open{background:#007aff20;color:#007aff}
.badge-mg{background:#d6336c20;color:#d6336c}
.badge-gray{background:#8e8e9320;color:#8e8e93}

/* Strategy cards */
.strat-card{background:#fff;border:1px solid #e5e5ea;border-radius:8px;padding:14px;position:relative}
.strat-card.active-border{border-left:3px solid #d6336c}
.strat-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}
.strat-name{font-weight:700;font-size:.85em}
.strat-arch{font-size:.7em;color:#d6336c;text-transform:uppercase;letter-spacing:.04em}
.strat-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:.75em}
.strat-stats .sl{color:#8e8e93}.strat-stats .sv{font-weight:600}
.cond-list{margin-top:6px;font-size:.7em;color:#636366}
.cond-list span{display:inline-block;background:#f2f2f7;padding:1px 6px;border-radius:4px;margin:1px 2px}

/* Kill bars */
.kill-bar-container{margin-top:8px}
.kill-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:.7em}
.kill-bar-label{width:90px;color:#8e8e93;flex-shrink:0;text-align:right}
.kill-bar-bg{flex:1;background:#f2f2f7;border-radius:4px;height:14px;overflow:hidden;position:relative}
.kill-bar-fill{height:100%;border-radius:4px 0 0 4px;transition:width .3s}
.kill-bar-val{width:50px;font-weight:600;flex-shrink:0}

/* Risk bars */
.risk-bar-bg{background:#2c2c2e;border-radius:8px;height:36px;position:relative;overflow:hidden;margin:8px 0 16px}
.risk-bar-fill{height:100%;border-radius:8px 0 0 8px;transition:width .3s}
.risk-bar-label{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:.75em;font-weight:700;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}
.risk-module{background:#fff;border:1px solid #e5e5ea;border-radius:10px;padding:14px;transition:border-color .15s}
.risk-module:hover{border-color:#d6336c}
.risk-module .rm-name{font-weight:700;font-size:.85em;margin-bottom:6px;color:#1d1d1f}
.risk-module .rm-arch{font-size:.7em;color:#8e8e93;margin-bottom:8px}
.risk-module .rm-row{display:flex;justify-content:space-between;font-size:.75em;padding:2px 0;border-bottom:1px solid #f2f2f7}
.risk-module .rm-row:last-child{border-bottom:none}
.risk-module .rm-row .rm-k{color:#8e8e93}
.risk-module .rm-row .rm-v{font-weight:600}

/* Gauge */
.gauge-container{display:flex;justify-content:center;margin:16px 0}
.gauge-svg{width:220px;height:130px}

/* Regime cards */
.regime-card{background:#fff;border:1px solid #e5e5ea;border-radius:8px;padding:14px;border-left:3px solid #8e8e93}
.regime-card.regime-win{border-left-color:#34c759}
.regime-card.regime-loss{border-left-color:#ff3b30}

/* Refresh bar */
.refresh-bar{display:flex;align-items:center;gap:12px;margin-bottom:12px;font-size:.75em;color:#8e8e93}

/* No data */
.no-data{text-align:center;padding:30px;color:#8e8e93;font-size:.9em}

/* Hidden */
.hidden{display:none}

/* Config display */
.config-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:.8em}
.config-row{display:flex;justify-content:space-between;padding:4px 8px;background:#f9f9fb;border-radius:4px}
.config-key{color:#8e8e93}
.config-val{font-weight:600}

/* Calendar */
.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:16px}
.cal-cell{border:1px solid #e5e5ea;border-radius:4px;padding:4px;font-size:.7em;position:relative;min-height:40px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.cal-cell.cal-green{background:#34c75915;border-color:#34c75940}
.cal-cell.cal-red{background:#ff3b3015;border-color:#ff3b3040}
.cal-cell.cal-grey{background:#f2f2f7;border-color:#e5e5ea;opacity:.4}
.cal-cell .cal-day{font-weight:700;font-size:.85em}
.cal-cell .cal-pnl{font-size:.7em;font-weight:600;margin-top:2px}
.cal-hdr{text-align:center;font-size:.65em;color:#8e8e93;text-transform:uppercase;letter-spacing:.06em;padding:6px 0}

/* Sortable table headers */
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:#d6336c}
th.sort-asc::after{content:' \u25B2';font-size:.7em}
th.sort-desc::after{content:' \u25BC';font-size:.7em}

/* Filter bar */
.filter-bar{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.filter-bar select,.filter-bar input{padding:5px 8px;border:1px solid #e5e5ea;border-radius:4px;font-family:inherit;font-size:.8em;background:#fff}
.filter-bar select:focus,.filter-bar input:focus{outline:none;border-color:#d6336c}
.filter-bar label{font-size:.75em;color:#8e8e93;text-transform:uppercase;letter-spacing:.04em}

/* Responsive */
@media(max-width:768px){.main{padding:12px}.grid{grid-template-columns:1fr}.m-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
"""


# ── Body / HTML structure ─────────────────────────────────────────────────────

_BODY = """<body>
<div class="hdr">
  <h1>OverfitAlpha <span>options trading console</span></h1>
  <div class="hdr-right">
    <span id="hdr-status"></span>
    <span id="hdr-clock"></span>
  </div>
</div>
<div class="tabs" id="tabBar">
  <div class="tab active" data-tab="dashboard">Dashboard</div>
  <div class="tab" data-tab="strategies">Strategies</div>
  <div class="tab" data-tab="risk">Risk</div>
  <div class="tab" data-tab="regimes">Regimes</div>
  <div class="tab" data-tab="trades">Trades</div>
  <div class="tab" data-tab="system">System</div>
</div>
<div class="main">
  <div id="tab-dashboard"></div>
  <div id="tab-strategies" class="hidden"></div>
  <div id="tab-risk" class="hidden"></div>
  <div id="tab-regimes" class="hidden"></div>
  <div id="tab-trades" class="hidden"></div>
  <div id="tab-system" class="hidden"></div>
</div>
"""


# ── JavaScript ────────────────────────────────────────────────────────────────

_JS = r"""<script>
// ── State ──
let D = null;
let currentTab = 'dashboard';
let _sortCol = null;
let _sortDir = 'asc';
let _filterDir = '';
let _filterStrat = '';

// ── Clock ──
setInterval(() => {
  const now = new Date();
  const el = document.getElementById('hdr-clock');
  if (el) el.textContent = now.toLocaleTimeString('en-US', {hour12: false, timeZone: 'America/New_York'}) + ' ET';
}, 1000);

// ── Tabs ──
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    currentTab = t.dataset.tab;
    document.querySelectorAll('[id^="tab-"]').forEach(x => x.classList.add('hidden'));
    document.getElementById('tab-' + currentTab).classList.remove('hidden');
    render();
  });
});

// ── Helpers ──
function fmt(n, d) { d = d != null ? d : 2; return n != null ? Number(n).toFixed(d) : '\u2014'; }
function fmtUSD(n) { return n != null ? '$' + Number(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '\u2014'; }
function pnlCls(v) { return v == null ? 'gray' : v >= 0 ? 'green' : 'red'; }
function pnlStr(v) { if (v == null) return '\u2014'; return (v >= 0 ? '+' : '') + fmtUSD(v); }
function pctStr(v) { if (v == null) return '\u2014'; return (v * 100).toFixed(1) + '%'; }
function esc(v) { if (v == null) return ''; return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function mc(label, value, cls) {
  return '<div class="m-card"><div class="m-label">' + label + '</div><div class="m-val ' + (cls || '') + '">' + value + '</div></div>';
}

// ── Fetch ──
let _refreshTimer = null;
async function fetchLive() {
  try {
    const r = await fetch('/api/options/live');
    if (!r.ok) throw new Error(r.statusText);
    D = await r.json();
    render();
  } catch(e) {
    document.getElementById('tab-dashboard').innerHTML = '<div class="no-data">Error loading data: ' + e.message + '</div>';
  }
}

function startRefresh() {
  fetchLive();
  if (_refreshTimer) clearInterval(_refreshTimer);
  _refreshTimer = setInterval(fetchLive, 15000);
}

function render() {
  if (!D) return;
  updateHeader();
  if (currentTab === 'dashboard') renderDashboard();
  else if (currentTab === 'strategies') renderStrategies();
  else if (currentTab === 'risk') renderRisk();
  else if (currentTab === 'regimes') renderRegimes();
  else if (currentTab === 'trades') renderTrades();
  else if (currentTab === 'system') renderSystem();
}

function updateHeader() {
  const el = document.getElementById('hdr-status');
  if (!el) return;
  const mode = D.session ? D.session.mode : 'livesim';
  if (mode === 'live') {
    el.innerHTML = '<span class="status-pill pill-active"><span class="hdr-dot dot-live"></span>LIVE</span>';
  } else {
    el.innerHTML = '<span class="status-pill pill-sim"><span class="hdr-dot" style="background:#007aff"></span>LIVESIM</span>';
  }
}

// ═══════════════════════════════════════════════════════════════════════
// TAB 1: Dashboard
// ═══════════════════════════════════════════════════════════════════════
function renderDashboard() {
  const el = document.getElementById('tab-dashboard');
  const strats = D.strategies || [];
  const active = strats.filter(s => s.status === 'active');
  const trades = D.trades || [];
  const equity = D.equity_curve || [];
  const recentTrades = trades.slice(-15).reverse();

  // Compute metrics
  const cumPnl = trades.reduce((a, t) => a + (t.scaled_pnl || 0), 0);
  const todayDate = (D.session || {}).date || '';
  const todayTrades = trades.filter(t => t.date === todayDate);
  const todayPnl = todayTrades.reduce((a, t) => a + (t.scaled_pnl || 0), 0);
  const totalTrades = trades.length;
  const wins = trades.filter(t => (t.scaled_pnl || 0) > 0).length;
  const wr = totalTrades > 0 ? (wins / totalTrades * 100).toFixed(1) + '%' : '\u2014';

  // Max drawdown
  let peak = 0, maxDD = 0;
  let cum = 0;
  for (const t of trades) {
    cum += (t.scaled_pnl || 0);
    if (cum > peak) peak = cum;
    const dd = peak - cum;
    if (dd > maxDD) maxDD = dd;
  }

  // Avg daily PnL
  const dailyMap = {};
  for (const t of trades) {
    if (!dailyMap[t.date]) dailyMap[t.date] = 0;
    dailyMap[t.date] += (t.scaled_pnl || 0);
  }
  const dailyVals = Object.values(dailyMap);
  const avgDailyPnl = dailyVals.length > 0 ? dailyVals.reduce((a, v) => a + v, 0) / dailyVals.length : 0;

  // Peak concurrent
  const peakConcurrent = (D.session || {}).peak_concurrent || active.length;

  let html = '<div class="refresh-bar"><span>Updated ' + new Date().toLocaleTimeString() + '</span></div>';

  // Metric cards
  html += '<div class="m-grid" style="margin-bottom:16px">';
  html += mc('Cumulative P&L', pnlStr(cumPnl), pnlCls(cumPnl));
  html += mc("Today's P&L", pnlStr(todayPnl), pnlCls(todayPnl));
  html += mc('Active Strategies', String(active.length), 'mg');
  html += mc('Total Trades', String(totalTrades), '');
  html += mc('Win Rate', wr, wins / Math.max(totalTrades, 1) >= 0.5 ? 'green' : 'red');
  html += mc('Max Drawdown', fmtUSD(maxDD), 'red');
  html += mc('Avg Daily P&L', pnlStr(avgDailyPnl), pnlCls(avgDailyPnl));
  html += mc('Peak Concurrent', String(peakConcurrent), 'blue');
  html += '</div>';

  // Equity curve canvas
  if (equity.length >= 2) {
    html += '<div class="card section"><h3>Equity Curve</h3>';
    html += '<canvas id="eqCanvas" style="width:100%;height:220px;border-radius:6px;cursor:crosshair"></canvas>';
    html += '<div id="eqTooltip" style="font-size:.75em;color:#8e8e93;margin-top:6px;min-height:1.5em"></div>';
    html += '</div>';
  }

  // Active strategies
  html += '<div class="section"><div class="section-title">Active Strategies</div>';
  html += '<div class="grid grid-wide">';
  for (const s of active) {
    const g = s.genes || {};
    html += '<div class="strat-card active-border">';
    html += '<div class="strat-header"><div>';
    html += '<div class="strat-name">' + esc(s.strategy_id || '?') + '</div>';
    html += '<div class="strat-arch">' + esc(g.archetype || '?') + ' / ' + esc((g.direction || '?')).toUpperCase() + '</div>';
    html += '</div><span class="badge badge-mg">LS:' + fmt(s.live_score, 2) + '</span></div>';
    html += '<div class="strat-stats">';
    html += '<div><span class="sl">Start</span> <span class="sv">' + esc(s.start_date) + '</span></div>';
    html += '<div><span class="sl">Trades</span> <span class="sv">' + (s.n_trades || 0) + '</span></div>';
    html += '<div><span class="sl">P&L</span> <span class="sv ' + pnlCls(s.cum_pnl) + '">' + pnlStr(s.cum_pnl) + '</span></div>';
    html += '<div><span class="sl">DD</span> <span class="sv red">' + fmtUSD(s.current_dd || 0) + '</span></div>';
    html += '<div><span class="sl">Contracts</span> <span class="sv">' + (s.contracts || 1) + '</span></div>';
    html += '<div><span class="sl">Live Score</span> <span class="sv mg">' + fmt(s.live_score, 2) + 'x</span></div>';
    html += '</div>';
    // Gene description
    html += '<div class="cond-list">';
    if (g.entry_conditions) {
      for (const c of g.entry_conditions) html += '<span>' + esc(c) + '</span>';
    }
    html += '</div>';
    html += '</div>';
  }
  html += '</div></div>';

  // Recent trades table
  if (recentTrades.length) {
    html += '<div class="section"><div class="section-title">Recent Trades</div>';
    html += '<div style="overflow-x:auto;max-height:400px;overflow-y:auto"><table><thead><tr>';
    html += '<th>Date</th><th>Strategy</th><th>Dir</th><th>Entry</th><th>Exit</th><th>Exit Reason</th><th>P&L</th><th>Cts</th>';
    html += '</tr></thead><tbody>';
    for (const t of recentTrades) {
      const pnl = t.scaled_pnl || 0;
      html += '<tr>';
      html += '<td>' + esc(t.date) + '</td>';
      html += '<td>' + esc(t.strategy_id || '\u2014') + '</td>';
      html += '<td style="font-weight:600">' + esc((t.direction || '').toUpperCase()) + '</td>';
      html += '<td>' + esc(t.entry_time || '\u2014') + '</td>';
      html += '<td>' + esc(t.exit_time || '\u2014') + '</td>';
      html += '<td style="font-size:.75em;color:#8e8e93">' + esc(t.exit_reason || '\u2014') + '</td>';
      html += '<td class="' + pnlCls(pnl) + ' pnl">' + pnlStr(pnl) + '</td>';
      html += '<td>' + (t.contracts || 1) + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table></div></div>';
  }

  el.innerHTML = html;

  // Draw equity curve
  if (equity.length >= 2) {
    setTimeout(() => { drawEquityCurve(equity); }, 50);
  }
}

function drawEquityCurve(equity) {
  const canvas = document.getElementById('eqCanvas');
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * (window.devicePixelRatio || 1);
  canvas.height = rect.height * (window.devicePixelRatio || 1);
  const ctx = canvas.getContext('2d');
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = rect.width, h = rect.height;
  const pad = {t: 20, r: 20, b: 30, l: 60};
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

  const vals = equity.map(e => e.cum_pnl);
  const minV = Math.min(0, ...vals);
  const maxV = Math.max(...vals) * 1.05 || 1;
  const range = maxV - minV || 1;

  // Background
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = '#e5e5ea';
  ctx.lineWidth = 0.5;
  const nGrid = 5;
  for (let i = 0; i <= nGrid; i++) {
    const y = pad.t + (ph / nGrid) * i;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    const val = maxV - (range / nGrid) * i;
    ctx.fillStyle = '#8e8e93';
    ctx.font = '10px SF Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('$' + val.toFixed(0), pad.l - 6, y + 4);
  }

  // Zero line
  const zeroY = pad.t + ((maxV - 0) / range) * ph;
  ctx.strokeStyle = '#1d1d1f40';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(w - pad.r, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Equity line
  ctx.strokeStyle = '#d6336c';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < vals.length; i++) {
    const x = pad.l + (i / (vals.length - 1)) * pw;
    const y = pad.t + ((maxV - vals[i]) / range) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill under curve
  const grad = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
  grad.addColorStop(0, 'rgba(214,51,108,0.15)');
  grad.addColorStop(1, 'rgba(214,51,108,0)');
  ctx.fillStyle = grad;
  ctx.lineTo(pad.l + pw, pad.t + ph);
  ctx.lineTo(pad.l, pad.t + ph);
  ctx.closePath();
  ctx.fill();

  // X-axis labels
  ctx.fillStyle = '#8e8e93';
  ctx.font = '10px SF Mono, monospace';
  ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(equity.length / 6));
  for (let i = 0; i < equity.length; i += step) {
    const x = pad.l + (i / (equity.length - 1)) * pw;
    ctx.fillText(equity[i].date.slice(5), x, h - 8);
  }

  // Tooltip on hover
  canvas.onmousemove = function(ev) {
    const br = canvas.getBoundingClientRect();
    const mx = ev.clientX - br.left;
    const idx = Math.round(((mx - pad.l) / pw) * (equity.length - 1));
    if (idx >= 0 && idx < equity.length) {
      const e = equity[idx];
      document.getElementById('eqTooltip').textContent = e.date + '  |  P&L: ' + pnlStr(e.cum_pnl) + '  |  Trades: ' + (e.n_trades || 0);
    }
  };
}

// ═══════════════════════════════════════════════════════════════════════
// TAB 2: Strategies
// ═══════════════════════════════════════════════════════════════════════
function renderStrategies() {
  const el = document.getElementById('tab-strategies');
  const strats = (D.strategies || []).filter(s => s.status === 'active');
  const killCfg = D.kill_config || {};

  if (!strats.length) { el.innerHTML = '<div class="no-data">No active strategies</div>'; return; }

  let html = '<div class="refresh-bar"><span>Active Strategy Pool &middot; ' + strats.length + ' strategies</span></div>';
  html += '<div class="grid grid-wide">';

  for (const s of strats) {
    const g = s.genes || {};
    const ts = s.training_stats || {};
    const fs = s.forward_stats || {};

    html += '<div class="strat-card active-border">';
    html += '<div class="strat-header"><div>';
    html += '<div class="strat-name">' + esc(s.strategy_id) + '</div>';
    html += '<div class="strat-arch">' + esc(g.archetype) + ' / ' + esc((g.direction || '').toUpperCase()) + '</div>';
    html += '</div><span class="badge badge-mg">LS:' + fmt(s.live_score, 2) + 'x</span></div>';

    // Genes detail
    html += '<div style="margin:8px 0;font-size:.72em;color:#636366">';
    html += '<div><b>Trade Type:</b> ' + esc(g.trade_type || 'options') + '</div>';
    html += '<div><b>Time Window:</b> ' + esc(g.time_window || '09:35-15:45') + '</div>';
    if (g.entry_conditions) {
      html += '<div style="margin-top:4px"><b>Entry:</b></div><div class="cond-list">';
      for (const c of g.entry_conditions) html += '<span>' + esc(c) + '</span>';
      html += '</div>';
    }
    if (g.exit_rules) {
      html += '<div style="margin-top:4px"><b>Exit:</b></div><div class="cond-list">';
      for (const c of g.exit_rules) html += '<span>' + esc(c) + '</span>';
      html += '</div>';
    }
    html += '</div>';

    // Training stats
    html += '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #f2f2f7">';
    html += '<div style="font-size:.65em;color:#8e8e93;text-transform:uppercase;margin-bottom:4px">Training</div>';
    html += '<div class="strat-stats">';
    html += '<div><span class="sl">Fitness</span> <span class="sv mg">' + fmt(ts.fitness, 3) + '</span></div>';
    html += '<div><span class="sl">P&L</span> <span class="sv ' + pnlCls(ts.pnl) + '">' + pnlStr(ts.pnl) + '</span></div>';
    html += '<div><span class="sl">WR</span> <span class="sv">' + (ts.wr != null ? (ts.wr * 100).toFixed(0) + '%' : '\u2014') + '</span></div>';
    html += '<div><span class="sl">Sharpe</span> <span class="sv">' + fmt(ts.sharpe, 2) + '</span></div>';
    html += '</div></div>';

    // Forward stats
    html += '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #f2f2f7">';
    html += '<div style="font-size:.65em;color:#8e8e93;text-transform:uppercase;margin-bottom:4px">Forward / Live</div>';
    html += '<div class="strat-stats">';
    html += '<div><span class="sl">Trades</span> <span class="sv">' + (fs.trades || s.n_trades || 0) + '</span></div>';
    html += '<div><span class="sl">P&L</span> <span class="sv ' + pnlCls(fs.pnl || s.cum_pnl) + '">' + pnlStr(fs.pnl || s.cum_pnl) + '</span></div>';
    html += '<div><span class="sl">WR</span> <span class="sv">' + (fs.wr != null ? (fs.wr * 100).toFixed(0) + '%' : '\u2014') + '</span></div>';
    html += '<div><span class="sl">DD</span> <span class="sv red">' + fmtUSD(fs.dd || s.current_dd || 0) + '</span></div>';
    html += '<div><span class="sl">Days</span> <span class="sv">' + (fs.days_alive || '\u2014') + '</span></div>';
    html += '<div><span class="sl">Contracts</span> <span class="sv">' + (s.contracts || 1) + '</span></div>';
    html += '</div></div>';

    // Kill condition bars
    html += renderKillBars(s, killCfg);

    html += '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function renderKillBars(s, kc) {
  if (!kc || Object.keys(kc).length === 0) return '';
  const fs = s.forward_stats || {};
  let html = '<div class="kill-bar-container" style="margin-top:8px;padding-top:8px;border-top:1px solid #f2f2f7">';
  html += '<div style="font-size:.65em;color:#8e8e93;text-transform:uppercase;margin-bottom:6px">Kill Thresholds</div>';

  // DD kill
  const ddMax = kc.max_dd_dollars || 3000;
  const ddCur = fs.dd || s.current_dd || 0;
  const ddPct = Math.min(ddCur / ddMax * 100, 100);
  html += killBarRow('Max DD', ddCur, ddMax, ddPct, '$');

  // Consecutive losses
  const clMax = kc.max_consec_losses || 5;
  const clCur = fs.consec_losses || 0;
  const clPct = Math.min(clCur / clMax * 100, 100);
  html += killBarRow('Consec L', clCur, clMax, clPct, '');

  // STL (short-term loss)
  const stlMax = kc.stl_max_loss || 800;
  const stlCur = fs.stl_loss || 0;
  const stlPct = Math.min(stlCur / stlMax * 100, 100);
  html += killBarRow('STL', stlCur, stlMax, stlPct, '$');

  // Days alive max
  const dayMax = kc.max_days_alive || 90;
  const dayCur = fs.days_alive || 0;
  const dayPct = Math.min(dayCur / dayMax * 100, 100);
  html += killBarRow('Age', dayCur, dayMax, dayPct, 'd');

  html += '</div>';
  return html;
}

function killBarRow(label, current, max, pct, unit) {
  const color = pct > 80 ? '#ff3b30' : pct > 50 ? '#ff9500' : '#34c759';
  let html = '<div class="kill-bar-row">';
  html += '<div class="kill-bar-label">' + label + '</div>';
  html += '<div class="kill-bar-bg"><div class="kill-bar-fill" style="width:' + pct.toFixed(1) + '%;background:' + color + '"></div></div>';
  html += '<div class="kill-bar-val">' + (unit === '$' ? fmtUSD(current) : current + unit) + '/' + (unit === '$' ? fmtUSD(max) : max + unit) + '</div>';
  html += '</div>';
  return html;
}

// ═══════════════════════════════════════════════════════════════════════
// TAB 3: Risk
// ═══════════════════════════════════════════════════════════════════════
function renderRisk() {
  const el = document.getElementById('tab-risk');
  const strats = (D.strategies || []).filter(s => s.status === 'active');
  const riskCfg = D.risk_config || {};
  const maxPortDD = riskCfg.max_portfolio_dd || 5000;
  const trades = D.trades || [];

  // Portfolio DD
  let portPeak = 0, portDD = 0, portCum = 0;
  for (const t of trades) {
    portCum += (t.scaled_pnl || 0);
    if (portCum > portPeak) portPeak = portCum;
    const dd = portPeak - portCum;
    if (dd > portDD) portDD = dd;
  }
  const portDDPct = Math.min(portDD / maxPortDD * 100, 100);

  let html = '<div class="refresh-bar"><span>Portfolio Risk &middot; Updated ' + new Date().toLocaleTimeString() + '</span></div>';

  // Portfolio DD gauge
  html += '<div class="card section">';
  html += '<h3>Portfolio Drawdown</h3>';
  html += '<div class="gauge-container"><svg class="gauge-svg" viewBox="0 0 220 130">';
  // Background arc
  html += '<path d="M 20 120 A 90 90 0 0 1 200 120" fill="none" stroke="#e5e5ea" stroke-width="16" stroke-linecap="round"/>';
  // Filled arc
  const angle = (portDDPct / 100) * Math.PI;
  const ex = 110 - 90 * Math.cos(angle);
  const ey = 120 - 90 * Math.sin(angle);
  const largeArc = portDDPct > 50 ? 1 : 0;
  const gaugeColor = portDDPct > 80 ? '#ff3b30' : portDDPct > 50 ? '#ff9500' : '#34c759';
  html += '<path d="M 20 120 A 90 90 0 ' + largeArc + ' 1 ' + ex.toFixed(1) + ' ' + ey.toFixed(1) + '" fill="none" stroke="' + gaugeColor + '" stroke-width="16" stroke-linecap="round"/>';
  html += '<text x="110" y="100" text-anchor="middle" font-size="22" font-weight="700" fill="#1d1d1f">' + portDDPct.toFixed(1) + '%</text>';
  html += '<text x="110" y="118" text-anchor="middle" font-size="10" fill="#8e8e93">' + fmtUSD(portDD) + ' / ' + fmtUSD(maxPortDD) + '</text>';
  html += '</svg></div></div>';

  // Per-strategy risk modules
  html += '<div class="section"><div class="section-title">Per-Strategy Risk</div>';
  html += '<div class="grid grid-wide">';

  const killCfg = D.kill_config || {};
  const baseDD = killCfg.max_dd_dollars || 3000;

  for (const s of strats) {
    const fs = s.forward_stats || {};
    const ddAllowance = baseDD * (s.live_score || 1) * (s.contracts || 1);
    const ddUsed = fs.dd || s.current_dd || 0;
    const ddPct = Math.min(ddUsed / ddAllowance * 100, 100);
    const ddColor = ddPct > 80 ? '#ff3b30' : ddPct > 50 ? '#ff9500' : '#34c759';

    const stlMax = killCfg.stl_max_loss || 800;
    const stlUsed = fs.stl_loss || 0;
    const stlPct = Math.min(stlUsed / stlMax * 100, 100);

    html += '<div class="risk-module">';
    html += '<div class="rm-name">' + esc(s.strategy_id) + '</div>';
    html += '<div class="rm-arch">' + esc((s.genes || {}).archetype || '?') + ' / ' + esc(((s.genes || {}).direction || '?').toUpperCase()) + '</div>';

    // DD bar
    html += '<div style="font-size:.7em;color:#8e8e93;margin-bottom:2px">DD: ' + fmtUSD(ddUsed) + ' / ' + fmtUSD(ddAllowance) + ' (base=' + fmtUSD(baseDD) + ' * LS=' + fmt(s.live_score, 2) + ' * cts=' + (s.contracts || 1) + ')</div>';
    html += '<div class="risk-bar-bg" style="height:20px;margin:4px 0 8px"><div class="risk-bar-fill" style="width:' + ddPct.toFixed(1) + '%;background:' + ddColor + '"></div></div>';

    // STL bar
    html += '<div style="font-size:.7em;color:#8e8e93;margin-bottom:2px">STL: ' + fmtUSD(stlUsed) + ' / ' + fmtUSD(stlMax) + '</div>';
    html += '<div class="risk-bar-bg" style="height:14px;margin:4px 0 8px"><div class="risk-bar-fill" style="width:' + stlPct.toFixed(1) + '%;background:' + (stlPct > 80 ? '#ff3b30' : '#007aff') + '"></div></div>';

    // Detail rows
    html += '<div class="rm-row"><span class="rm-k">Contracts</span><span class="rm-v">' + (s.contracts || 1) + '</span></div>';
    html += '<div class="rm-row"><span class="rm-k">Live Score</span><span class="rm-v mg">' + fmt(s.live_score, 2) + 'x</span></div>';
    html += '<div class="rm-row"><span class="rm-k">P&L</span><span class="rm-v ' + pnlCls(s.cum_pnl) + '">' + pnlStr(s.cum_pnl) + '</span></div>';
    html += '<div class="rm-row"><span class="rm-k">Peak Equity</span><span class="rm-v">' + fmtUSD(fs.peak_equity || 0) + '</span></div>';
    html += '</div>';
  }
  html += '</div></div>';

  // Contract scaling visualization
  html += '<div class="card section"><h3>Contract Scaling Formula</h3>';
  html += '<div style="font-size:.8em;color:#636366;line-height:1.6">';
  html += '<code>contracts = clamp(floor(live_score / scale_threshold), 1, max_contracts)</code><br>';
  html += '<div style="margin-top:6px"><b>scale_threshold:</b> ' + (riskCfg.scale_threshold || 1.5) + ' &nbsp; <b>max_contracts:</b> ' + (riskCfg.max_contracts || 5) + '</div>';
  html += '<div style="margin-top:8px">';
  html += '<table><thead><tr><th>Strategy</th><th>Live Score</th><th>Contracts</th></tr></thead><tbody>';
  for (const s of strats) {
    html += '<tr><td>' + esc(s.strategy_id) + '</td><td>' + fmt(s.live_score, 2) + '</td><td style="font-weight:700">' + (s.contracts || 1) + '</td></tr>';
  }
  html += '</tbody></table></div></div></div>';

  el.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════
// TAB 4: Regimes
// ═══════════════════════════════════════════════════════════════════════
function renderRegimes() {
  const el = document.getElementById('tab-regimes');
  const regimes = D.completed_regimes || [];

  if (!regimes.length) { el.innerHTML = '<div class="no-data">No completed regimes</div>'; return; }

  const winners = regimes.filter(r => (r.pnl || 0) > 0);
  const losers = regimes.filter(r => (r.pnl || 0) <= 0);

  let html = '<div class="refresh-bar"><span>Completed Regimes &middot; ' + regimes.length + ' total</span></div>';

  // Summary
  html += '<div class="m-grid" style="margin-bottom:16px">';
  html += mc('Total Regimes', String(regimes.length), '');
  html += mc('Winners', String(winners.length), 'green');
  html += mc('Losers', String(losers.length), 'red');
  html += mc('Win Rate', regimes.length > 0 ? (winners.length / regimes.length * 100).toFixed(0) + '%' : '\u2014', winners.length >= losers.length ? 'green' : 'red');
  html += '</div>';

  // Death reason distribution
  const deathReasons = {};
  for (const r of regimes) {
    const reason = r.death_reason || 'unknown';
    deathReasons[reason] = (deathReasons[reason] || 0) + 1;
  }
  html += '<div class="card section"><h3>Death Reason Distribution</h3>';
  html += '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">';
  const barColors = ['#d6336c', '#007aff', '#ff9500', '#34c759', '#af52de', '#ff3b30', '#5ac8fa', '#ffcc00'];
  let ci = 0;
  for (const [reason, count] of Object.entries(deathReasons)) {
    const pct = (count / regimes.length * 100).toFixed(0);
    html += '<div style="flex:1;min-width:120px">';
    html += '<div style="font-size:.75em;color:#8e8e93;margin-bottom:4px">' + esc(reason) + '</div>';
    html += '<div style="background:#f2f2f7;border-radius:4px;height:24px;overflow:hidden">';
    html += '<div style="height:100%;width:' + pct + '%;background:' + barColors[ci % barColors.length] + ';border-radius:4px;display:flex;align-items:center;justify-content:center">';
    html += '<span style="font-size:.7em;font-weight:700;color:#fff">' + count + '</span>';
    html += '</div></div></div>';
    ci++;
  }
  html += '</div></div>';

  // Regime table
  html += '<div class="section"><div class="section-title">Regime History</div>';
  html += '<div style="overflow-x:auto"><table><thead><tr>';
  html += '<th>ID</th><th>Train Period</th><th>Forward Period</th><th>Days</th><th>Trades</th><th>P&L</th><th>WR</th><th>Max DD</th><th>Max Cts</th><th>Death Reason</th>';
  html += '</tr></thead><tbody>';
  for (const r of regimes) {
    const pnl = r.pnl || 0;
    const isWin = pnl > 0;
    html += '<tr>';
    html += '<td style="font-weight:600">' + esc(r.regime_id) + '</td>';
    html += '<td style="font-size:.75em">' + esc(r.train_period) + '</td>';
    html += '<td style="font-size:.75em">' + esc(r.forward_period) + '</td>';
    html += '<td>' + (r.days_alive || 0) + '</td>';
    html += '<td>' + (r.trades || 0) + '</td>';
    html += '<td class="' + pnlCls(pnl) + ' pnl">' + pnlStr(pnl) + '</td>';
    html += '<td>' + (r.wr != null ? (r.wr * 100).toFixed(0) + '%' : '\u2014') + '</td>';
    html += '<td class="red">' + fmtUSD(r.max_dd || 0) + '</td>';
    html += '<td>' + (r.max_contracts || 1) + '</td>';
    html += '<td><span class="badge ' + (isWin ? 'badge-win' : 'badge-loss') + '">' + esc(r.death_reason || '?') + '</span></td>';
    html += '</tr>';
  }
  html += '</tbody></table></div></div>';

  // Winners vs Losers cards
  html += '<div class="section"><div class="section-title">Winners vs Losers</div>';
  html += '<div class="grid">';
  for (const r of regimes) {
    const pnl = r.pnl || 0;
    const cls = pnl > 0 ? 'regime-win' : 'regime-loss';
    html += '<div class="regime-card ' + cls + '">';
    html += '<div style="font-weight:700;font-size:.85em">' + esc(r.regime_id) + '</div>';
    html += '<div style="font-size:.7em;color:#8e8e93;margin:4px 0">' + (r.days_alive || 0) + ' days &middot; ' + (r.trades || 0) + ' trades</div>';
    html += '<div class="' + pnlCls(pnl) + '" style="font-size:1.1em;font-weight:700">' + pnlStr(pnl) + '</div>';
    html += '<div style="font-size:.7em;color:#8e8e93;margin-top:4px">' + esc(r.death_reason) + '</div>';
    html += '</div>';
  }
  html += '</div></div>';

  el.innerHTML = html;
}

// ═══════════════════════════════════════════════════════════════════════
// TAB 5: Trades
// ═══════════════════════════════════════════════════════════════════════
function renderTrades() {
  const el = document.getElementById('tab-trades');
  let trades = (D.trades || []).slice();

  // Filters
  const directions = [...new Set(trades.map(t => t.direction).filter(Boolean))];
  const stratIds = [...new Set(trades.map(t => t.strategy_id).filter(Boolean))];

  let html = '<div class="refresh-bar"><span>Full Trade Log &middot; ' + trades.length + ' trades</span></div>';

  // Filter bar
  html += '<div class="filter-bar">';
  html += '<label>Direction</label><select id="fDir" onchange="_filterDir=this.value;renderTrades()">';
  html += '<option value="">All</option>';
  for (const d of directions) html += '<option value="' + d + '"' + (_filterDir === d ? ' selected' : '') + '>' + d.toUpperCase() + '</option>';
  html += '</select>';
  html += '<label>Strategy</label><select id="fStrat" onchange="_filterStrat=this.value;renderTrades()">';
  html += '<option value="">All</option>';
  for (const s of stratIds) html += '<option value="' + s + '"' + (_filterStrat === s ? ' selected' : '') + '>' + esc(s) + '</option>';
  html += '</select>';
  html += '</div>';

  // Apply filters
  if (_filterDir) trades = trades.filter(t => t.direction === _filterDir);
  if (_filterStrat) trades = trades.filter(t => t.strategy_id === _filterStrat);

  // Apply sort
  if (_sortCol != null) {
    const cols = ['date', 'strategy_id', 'direction', 'trade_type', 'entry_time', 'exit_time', 'exit_reason', 'pnl_per_ct', 'contracts', 'scaled_pnl', 'cum_pnl'];
    const key = cols[_sortCol];
    trades.sort((a, b) => {
      let va = a[key], vb = b[key];
      if (typeof va === 'number' && typeof vb === 'number') return _sortDir === 'asc' ? va - vb : vb - va;
      va = String(va || ''); vb = String(vb || '');
      return _sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }

  // Table
  html += '<div style="overflow-x:auto;max-height:600px;overflow-y:auto"><table><thead><tr>';
  const headers = ['Date', 'Strategy', 'Dir', 'Type', 'Entry', 'Exit', 'Exit Reason', 'P&L/ct', 'Cts', 'Scaled P&L', 'Cum P&L'];
  for (let i = 0; i < headers.length; i++) {
    const sortCls = _sortCol === i ? (_sortDir === 'asc' ? ' sort-asc' : ' sort-desc') : '';
    html += '<th class="sortable' + sortCls + '" onclick="sortTrades(' + i + ')">' + headers[i] + '</th>';
  }
  html += '</tr></thead><tbody>';

  for (const t of trades) {
    const pnl = t.scaled_pnl || 0;
    html += '<tr>';
    html += '<td>' + esc(t.date) + '</td>';
    html += '<td style="font-size:.75em">' + esc(t.strategy_id) + '</td>';
    html += '<td style="font-weight:600">' + esc((t.direction || '').toUpperCase()) + '</td>';
    html += '<td>' + esc(t.trade_type || 'options') + '</td>';
    html += '<td>' + esc(t.entry_time || '\u2014') + '</td>';
    html += '<td>' + esc(t.exit_time || '\u2014') + '</td>';
    html += '<td style="font-size:.75em;color:#8e8e93">' + esc(t.exit_reason || '\u2014') + '</td>';
    html += '<td class="' + pnlCls(t.pnl_per_ct) + ' pnl">' + pnlStr(t.pnl_per_ct) + '</td>';
    html += '<td>' + (t.contracts || 1) + '</td>';
    html += '<td class="' + pnlCls(pnl) + ' pnl">' + pnlStr(pnl) + '</td>';
    html += '<td class="' + pnlCls(t.cum_pnl) + ' pnl">' + pnlStr(t.cum_pnl) + '</td>';
    html += '</tr>';
  }
  html += '</tbody></table></div>';

  el.innerHTML = html;
}

function sortTrades(col) {
  if (_sortCol === col) {
    _sortDir = _sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    _sortCol = col;
    _sortDir = 'asc';
  }
  renderTrades();
}

// ═══════════════════════════════════════════════════════════════════════
// TAB 6: System
// ═══════════════════════════════════════════════════════════════════════
function renderSystem() {
  const el = document.getElementById('tab-system');
  const kc = D.kill_config || {};
  const rc = D.risk_config || {};
  const ls = D.live_score_config || {};
  const build = D.build_info || {};

  let html = '<div class="refresh-bar"><span>System Configuration</span></div>';

  // KillConfig
  html += '<div class="card section"><h3>Kill Configuration</h3>';
  html += '<div class="config-grid">';
  for (const [k, v] of Object.entries(kc)) {
    html += '<div class="config-row"><span class="config-key">' + esc(k) + '</span><span class="config-val">' + esc(String(v)) + '</span></div>';
  }
  html += '</div></div>';

  // Live score formula
  html += '<div class="card section"><h3>Live Score Formula</h3>';
  html += '<div style="font-size:.82em;color:#636366;line-height:1.7">';
  html += '<code>live_score = base_score</code><br>';
  html += '<code>&nbsp;&nbsp;+ wr_bonus * max(0, forward_wr - ' + (ls.wr_threshold || 0.5) + ')</code><br>';
  html += '<code>&nbsp;&nbsp;+ pnl_bonus * max(0, forward_pnl / ' + (ls.pnl_norm || 500) + ')</code><br>';
  html += '<code>&nbsp;&nbsp;+ streak_bonus * consec_wins</code><br>';
  html += '<code>&nbsp;&nbsp;- dd_penalty * (current_dd / max_dd)</code><br>';
  html += '<div style="margin-top:8px">';
  html += '<b>base_score:</b> ' + (ls.base_score || 1.0) + ' &nbsp; ';
  html += '<b>wr_bonus:</b> ' + (ls.wr_bonus || 0.3) + ' &nbsp; ';
  html += '<b>pnl_bonus:</b> ' + (ls.pnl_bonus || 0.2) + ' &nbsp; ';
  html += '<b>streak_bonus:</b> ' + (ls.streak_bonus || 0.1) + ' &nbsp; ';
  html += '<b>dd_penalty:</b> ' + (ls.dd_penalty || 0.5) + '</div>';
  html += '</div></div>';

  // Contract scaling config
  html += '<div class="card section"><h3>Contract Scaling</h3>';
  html += '<div class="config-grid">';
  for (const [k, v] of Object.entries(rc)) {
    html += '<div class="config-row"><span class="config-key">' + esc(k) + '</span><span class="config-val">' + esc(String(v)) + '</span></div>';
  }
  html += '</div></div>';

  // Build info
  html += '<div class="card section"><h3>Build Information</h3>';
  html += '<div class="config-grid">';
  for (const [k, v] of Object.entries(build)) {
    html += '<div class="config-row"><span class="config-key">' + esc(k) + '</span><span class="config-val" style="font-size:.8em;word-break:break-all">' + esc(String(v)) + '</span></div>';
  }
  html += '</div></div>';

  // Data availability calendar
  html += '<div class="card section"><h3>Data Availability</h3>';
  const cal = D.data_calendar || [];
  if (cal.length) {
    html += '<div class="cal-grid">';
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    for (const d of dayNames) html += '<div class="cal-hdr">' + d + '</div>';

    // Pad to start on correct weekday
    if (cal.length > 0) {
      const firstDay = new Date(cal[0].date + 'T12:00:00').getDay();
      for (let i = 0; i < firstDay; i++) html += '<div class="cal-cell cal-grey"><div class="cal-day"></div></div>';
    }

    for (const c of cal) {
      const cls = c.has_data ? (c.pnl > 0 ? 'cal-green' : c.pnl < 0 ? 'cal-red' : '') : 'cal-grey';
      html += '<div class="cal-cell ' + cls + '">';
      html += '<div class="cal-day">' + c.date.slice(8) + '</div>';
      if (c.has_data && c.pnl != null) {
        html += '<div class="cal-pnl ' + pnlCls(c.pnl) + '">' + (c.pnl >= 0 ? '+' : '') + '$' + Math.abs(c.pnl).toFixed(0) + '</div>';
      }
      html += '</div>';
    }
    html += '</div>';
  } else {
    html += '<div class="no-data">No calendar data</div>';
  }
  html += '</div>';

  el.innerHTML = html;
}

// ── Startup ──
startRefresh();
</script>
"""
