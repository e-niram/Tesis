/**
 * app.js — Main controller for the Madrid Noise Forecast dashboard.
 *
 * Responsibilities:
 *   1. Load predictions.json (generated daily by the GitHub Actions pipeline)
 *   2. Wire up controls (period toggle, day slider)
 *   3. Initialise and coordinate the map (map.js) and chart (charts.js)
 *   4. Render the ranking table
 *   5. Update all UI on control changes
 */

'use strict';

/* ── State ──────────────────────────────────────────────────────────────── */
const state = {
  data:            null,    // full predictions.json payload
  period:          'daytime',
  dayIndex:        0,       // 0-based (0 = first forecast day = tomorrow)
  selectedStation: null,
};

/* ── DOM refs ───────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const DOM = {
  loading:          $('loading'),
  error:            $('data-error'),
  dashboard:        $('dashboard'),
  tableSection:     $('table-section'),
  lastUpdated:      $('last-updated'),
  footerUpdated:    $('footer-updated'),
  daySlider:        $('day-slider'),
  dayLabel:         $('day-label'),
  mapDateLabel:     $('map-date-label'),
  stationName:      $('selected-station-name'),
  chartHint:        $('chart-hint'),
  chartWrapper:     document.querySelector('.chart-wrapper'),
  stationStats:     $('station-stats'),
  statClusterDay:   $('stat-cluster-day'),
  statClusterNight: $('stat-cluster-night'),
  statMax:          $('stat-max'),
  statMin:          $('stat-min'),
  rankingBody:      $('ranking-body'),
  pagePrev:         $('page-prev'),
  pageNext:         $('page-next'),
  pageInfo:         $('page-info'),
  periodRadios:     document.querySelectorAll('input[name="period"]'),
};

/* ── Helpers ────────────────────────────────────────────────────────────── */
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true;  }

function clusterLabel(n) {
  const labels = [I18n.t('cluster.high'), I18n.t('cluster.low'), I18n.t('cluster.mid')];
  return labels[n] ?? `${I18n.t('cluster.n')} ${n}`;
}

function applyNoiseColor(el, laeq) {
  el.classList.remove('noise-low', 'noise-mid', 'noise-high');
  if (laeq === null) return;
  el.classList.add(`noise-${NoiseMap.noiseLevel(laeq).cssClass}`);
}

function forecastForStation(station) {
  const key = state.period === 'daytime' ? 'daytime_forecast' : 'nighttime_forecast';
  return station[key] || [];
}

function laeqAtDay(station) {
  const fc = forecastForStation(station);
  if (!fc.length) return null;
  return fc[Math.min(state.dayIndex, fc.length - 1)].laeq;
}

function dateAtDay(station) {
  const fc = forecastForStation(station);
  if (!fc.length) return '—';
  return fc[Math.min(state.dayIndex, fc.length - 1)].date;
}

function formatDateLong(iso) {
  if (!iso || iso === '—') return '—';
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString(I18n.t('date.locale'), { weekday:'short', day:'numeric', month:'long' });
}

/* ── Ranking table ──────────────────────────────────────────────────────── */
const tbl = { rows: [], page: 0, perPage: 10, sortCol: 'laeq', sortDir: -1 };

function renderTable() {
  tbl.rows = state.data.stations
    .map(s => ({ ...s, laeq: laeqAtDay(s) }))
    .filter(s => s.laeq !== null);
  tbl.page = 0;
  _sortRows();
  _renderPage();
  show(DOM.tableSection);
}

function _sortRows() {
  const { sortCol, sortDir } = tbl;
  tbl.rows.sort((a, b) => {
    const av = sortCol === 'name' ? a.name.toLowerCase() : a.laeq;
    const bv = sortCol === 'name' ? b.name.toLowerCase() : b.laeq;
    return sortDir * (av > bv ? 1 : av < bv ? -1 : 0);
  });
}

function _renderPage() {
  const start = tbl.page * tbl.perPage;
  const total = Math.ceil(tbl.rows.length / tbl.perPage);

  DOM.rankingBody.innerHTML = tbl.rows.slice(start, start + tbl.perPage).map((s, i) => {
    const { label, cssClass } = NoiseMap.noiseLevel(s.laeq);
    return `<tr>
      <td>${start + i + 1}</td>
      <td>${_esc(s.name)}</td>
      <td>${s.laeq.toFixed(1)} dB</td>
      <td><span class="badge badge-${cssClass}">${_esc(label)}</span></td>
    </tr>`;
  }).join('');

  DOM.pageInfo.textContent = `${tbl.page + 1} / ${total}`;
  DOM.pagePrev.disabled    = tbl.page === 0;
  DOM.pageNext.disabled    = tbl.page >= total - 1;

  document.querySelectorAll('.noise-table th[data-sort]').forEach(th => {
    th.classList.remove('sort-asc', 'sort-desc');
    if (th.dataset.sort === tbl.sortCol) {
      th.classList.add(tbl.sortDir === 1 ? 'sort-asc' : 'sort-desc');
    }
  });
}

/* ── Station detail (chart + stats) ────────────────────────────────────── */
function showStationDetail(station) {
  state.selectedStation = station;

  // Chart
  hide(DOM.chartHint);
  show(DOM.chartWrapper);
  NoiseCharts.updateChart(station, state.period);

  // Station name heading
  DOM.stationName.textContent = station.name;

  // Stats panel
  const fc    = forecastForStation(station);
  const vals  = fc.map(e => e.laeq);
  const maxDb = vals.length ? Math.max(...vals).toFixed(1) + ' dB' : '—';
  const minDb = vals.length ? Math.min(...vals).toFixed(1) + ' dB' : '—';

  DOM.statClusterDay.textContent   = clusterLabel(station.cluster_day);
  DOM.statClusterNight.textContent = clusterLabel(station.cluster_night);
  DOM.statMax.textContent          = maxDb;
  DOM.statMin.textContent          = minDb;
  applyNoiseColor(DOM.statMax, vals.length ? Math.max(...vals) : null);
  applyNoiseColor(DOM.statMin, vals.length ? Math.min(...vals) : null);
  show(DOM.stationStats);
}

/* ── Map + labels refresh ───────────────────────────────────────────────── */
function refreshMap() {
  const { stations } = state.data;
  NoiseMap.updateMapMarkers(stations, state.period, state.dayIndex);

  // Update the date label above the map
  const sampleDate = stations[0] ? dateAtDay(stations[0]) : '';
  DOM.mapDateLabel.textContent = sampleDate ? `— ${formatDateLong(sampleDate)}` : '';
}

/* ── Control handlers ───────────────────────────────────────────────────── */
function onPeriodChange(e) {
  state.period = e.target.value;
  refreshMap();
  renderTable();
  if (state.selectedStation) {
    showStationDetail(state.selectedStation);
  }
}

function onDayChange() {
  const val = parseInt(DOM.daySlider.value, 10);
  state.dayIndex = val - 1;
  DOM.dayLabel.textContent = `+${val}`;
  DOM.daySlider.setAttribute('aria-valuenow', val);
  refreshMap();
  renderTable();
}

/* ── Bootstrap ──────────────────────────────────────────────────────────── */
async function bootstrap() {
  try {
    const resp = await fetch('data/predictions.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.data = await resp.json();
  } catch (err) {
    console.error('Failed to load predictions.json:', err);
    hide(DOM.loading);
    show(DOM.error);
    return;
  }

  // Update timestamps
  const updated = state.data.last_updated || '';
  DOM.lastUpdated.textContent   = updated ? formatDateLong(updated) : '—';
  DOM.footerUpdated.textContent = updated ? `${I18n.t('footer.updated')} ${updated}` : '';

  // Init map
  const map = NoiseMap.initMap(station => showStationDetail(station));
  void map; // map instance stored internally in map.js

  // Init chart (empty)
  NoiseCharts.initChart();

  // Wire controls
  DOM.periodRadios.forEach(r => r.addEventListener('change', onPeriodChange));
  DOM.daySlider.addEventListener('input', onDayChange);

  // Table sorting
  document.querySelectorAll('.noise-table th[data-sort]').forEach(th => {
    th.addEventListener('click', () => {
      if (tbl.sortCol === th.dataset.sort) {
        tbl.sortDir *= -1;
      } else {
        tbl.sortCol = th.dataset.sort;
        tbl.sortDir = th.dataset.sort === 'name' ? 1 : -1;
      }
      tbl.page = 0;
      _sortRows();
      _renderPage();
    });
  });

  // Table pagination
  DOM.pagePrev.addEventListener('click', () => { tbl.page--; _renderPage(); });
  DOM.pageNext.addEventListener('click', () => { tbl.page++; _renderPage(); });

  // Info-icon tap support (mobile: click toggles popover; click elsewhere closes)
  document.querySelectorAll('.info-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const wrap = btn.closest('.info-wrap');
      const wasActive = wrap.classList.contains('active');
      document.querySelectorAll('.info-wrap.active').forEach(w => w.classList.remove('active'));
      if (!wasActive) wrap.classList.add('active');
    });
  });
  document.addEventListener('click', () => {
    document.querySelectorAll('.info-wrap.active').forEach(w => w.classList.remove('active'));
  });

  // Set translated station placeholder
  DOM.stationName.textContent = I18n.t('chart.station.placeholder');

  // Re-render dynamic content on language change
  document.addEventListener('langchange', () => {
    if (!state.data) return;
    const updated = state.data.last_updated || '';
    DOM.lastUpdated.textContent   = updated ? formatDateLong(updated) : '—';
    DOM.footerUpdated.textContent = updated ? `${I18n.t('footer.updated')} ${updated}` : '';
    refreshMap();
    renderTable();
    if (state.selectedStation) {
      showStationDetail(state.selectedStation);
    } else {
      DOM.stationName.textContent = I18n.t('chart.station.placeholder');
    }
  });

  // First render
  hide(DOM.loading);
  show(DOM.dashboard);
  refreshMap();
  renderTable();
}

/* ── Security helpers ───────────────────────────────────────────────────── */
function _esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── Start ──────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', bootstrap);
