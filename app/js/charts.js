/**
 * charts.js — Chart.js time series panel for the Madrid Noise Forecast dashboard.
 *
 * Exports (via window.NoiseCharts):
 *   initChart()
 *   updateChart(station, period)
 */

'use strict';

let _chart = null;

/* ── Colour palette matching style.css ─────────────────────────────────── */
const PALETTE = {
  daytime:  { line: '#003082' },
  nighttime:{ line: '#c8102e' },
};

function noiseColor(v) {
  if (v < 55) return '#2e7d32';
  if (v < 65) return '#e65100';
  return '#c62828';
}

/* ── WHO threshold annotation bands ────────────────────────────────────── */
// Rendered as horizontal reference lines via Chart.js annotation plugin.
// We draw them manually as dataset lines to avoid needing an extra CDN dep.
const WHO_LINES = [
  { y: 55, label: 'OMS: Moderado (55 dB)', color: 'rgba(230,81,0,.5)',  dash: [6,4] },
  { y: 65, label: 'OMS: Alto (65 dB)',     color: 'rgba(198,40,40,.5)', dash: [4,4] },
];

/* ── Format helpers ─────────────────────────────────────────────────────── */
function formatDate(iso) {
  // "2025-04-13" → "13 abr"
  const [, m, d] = iso.split('-');
  const months = ['','ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'];
  return `${parseInt(d)} ${months[parseInt(m)]}`;
}

/* ── Selected-day vertical line plugin ──────────────────────────────────── */
const selectedDayPlugin = {
  id: 'selectedDay',
  afterDraw(chart) {
    const idx = chart.options.plugins.selectedDay?.index;
    if (idx == null) return;
    const meta = chart.getDatasetMeta(0);
    if (!meta?.data[idx]) return;

    const x = meta.data[idx].x;
    const { ctx, chartArea: { top, bottom } } = chart;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(0,48,130,.55)';
    ctx.setLineDash([5, 4]);
    ctx.stroke();

    // Small label at the top of the line
    ctx.font = "600 10px 'Source Sans Pro', sans-serif";
    ctx.fillStyle = 'rgba(0,48,130,.8)';
    ctx.textAlign = 'center';
    ctx.fillText('▼ mapa', x, top - 2);
    ctx.restore();
  },
};
Chart.register(selectedDayPlugin);

/* ── Init ───────────────────────────────────────────────────────────────── */
function initChart() {
  const ctx = document.getElementById('forecast-chart').getContext('2d');

  _chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        selectedDay: { index: 0 },
        legend: {
          display: true,
          position: 'top',
          labels: {
            font: { family: "'Source Sans Pro', sans-serif", size: 12 },
            usePointStyle: true,
            pointStyleWidth: 10,
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} dB`,
          },
          bodyFont: { family: "'Source Sans Pro', sans-serif" },
          titleFont: { family: "'Source Sans Pro', sans-serif", weight: '600' },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(0,0,0,.06)' },
          ticks: {
            font: { family: "'Source Sans Pro', sans-serif", size: 11 },
            maxTicksLimit: 14,
          },
        },
        y: {
          title: {
            display: true,
            text: 'Nivel LAeq (dB)',
            font: { family: "'Source Sans Pro', sans-serif", size: 12 },
            color: '#555',
          },
          grid: { color: 'rgba(0,0,0,.06)' },
          ticks: {
            font: { family: "'Source Sans Pro', sans-serif", size: 11 },
            callback: (v) => `${v} dB`,
          },
        },
      },
    },
  });

  return _chart;
}

/* ── Update ─────────────────────────────────────────────────────────────── */

/**
 * Render the 14-day forecast for one station.
 *
 * @param {Object} station   Station entry from predictions.json
 * @param {string} period    'daytime' | 'nighttime'
 */
function updateChart(station, period, dayIndex) {
  if (!_chart) initChart();

  const forecastKey = period === 'daytime' ? 'daytime_forecast' : 'nighttime_forecast';
  const forecast    = station[forecastKey] || [];

  const labels  = forecast.map(e => formatDate(e.date));
  const values  = forecast.map(e => e.laeq);
  const pal     = PALETTE[period];
  const periodLabel = period === 'daytime' ? 'Diurno' : 'Nocturno';

  // Build WHO horizontal reference datasets (one point per day so they span full x-axis)
  const whoDatasets = WHO_LINES.map(ref => ({
    label: ref.label,
    data:  labels.map(() => ref.y),
    borderColor: ref.color,
    borderWidth: 1.5,
    borderDash: ref.dash,
    pointRadius: 0,
    fill: false,
    tension: 0,
    order: 10,   // draw behind the main line
  }));

  const pointColors = values.map(noiseColor);

  _chart.data.labels = labels;
  _chart.data.datasets = [
    {
      label: `Predicción (${periodLabel})`,
      data: values,
      borderColor: pal.line,       // legend swatch colour
      backgroundColor: 'rgba(0,0,0,.04)',
      borderWidth: 2.5,
      pointRadius: 4,
      pointHoverRadius: 6,
      pointBackgroundColor: pointColors,
      pointBorderColor: pointColors,
      fill: true,
      tension: 0.3,
      order: 1,
      segment: {
        borderColor: ctx => noiseColor(values[ctx.p0DataIndex]),
      },
    },
    ...whoDatasets,
  ];

  // Dynamic y-axis padding around data range
  const minVal = Math.min(...values, 45);
  const maxVal = Math.max(...values, 70);
  _chart.options.scales.y.min = Math.floor(minVal - 3);
  _chart.options.scales.y.max = Math.ceil(maxVal + 3);

  // Sync vertical line with the selected day on the map
  _chart.options.plugins.selectedDay.index = Math.min(dayIndex ?? 0, values.length - 1);

  _chart.update('active');
}

/* ── Public API ─────────────────────────────────────────────────────────── */
window.NoiseCharts = { initChart, updateChart };
