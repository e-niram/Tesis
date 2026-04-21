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
  periodRadios:     document.querySelectorAll('input[name="period"]'),
};

/* ── Helpers ────────────────────────────────────────────────────────────── */
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true;  }

const CLUSTER_LABELS = ['Ruido Bajo', 'Ruido Medio', 'Ruido Alto'];
function clusterLabel(n) {
  return CLUSTER_LABELS[n] ?? `Clúster ${n}`;
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
  return d.toLocaleDateString('es-ES', { weekday:'short', day:'numeric', month:'long' });
}

/* ── Ranking table ──────────────────────────────────────────────────────── */
function renderTable() {
  const { stations } = state.data;

  const sorted = stations
    .map(s => ({ ...s, laeq: laeqAtDay(s) }))
    .filter(s => s.laeq !== null)
    .sort((a, b) => b.laeq - a.laeq)
    .slice(0, 10);

  DOM.rankingBody.innerHTML = sorted.map((s, i) => {
    const { label, cssClass } = NoiseMap.noiseLevel(s.laeq);
    return `<tr>
      <td>${i + 1}</td>
      <td>${_esc(s.name)}</td>
      <td>${s.laeq.toFixed(1)} dB</td>
      <td><span class="badge badge-${cssClass}">${_esc(label)}</span></td>
    </tr>`;
  }).join('');

  show(DOM.tableSection);
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
  if (state.selectedStation) {
    // Refresh chart to highlight the selected day visually (chart shows all 14, map updates)
    NoiseCharts.updateChart(state.selectedStation, state.period);
  }
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
  DOM.footerUpdated.textContent = updated ? `Actualizado: ${updated}` : '';

  // Init map
  const map = NoiseMap.initMap(station => showStationDetail(station));
  void map; // map instance stored internally in map.js

  // Init chart (empty)
  NoiseCharts.initChart();

  // Wire controls
  DOM.periodRadios.forEach(r => r.addEventListener('change', onPeriodChange));
  DOM.daySlider.addEventListener('input', onDayChange);

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
