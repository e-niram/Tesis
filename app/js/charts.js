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
  if (v < 65) return '#ca8a04';
  return '#c62828';
}

/* ── WHO threshold annotation bands ────────────────────────────────────── */
function getWhoLines() {
  return [
    { y: 55, label: I18n.t('chart.who.mid'),  color: 'rgba(230,81,0,.5)',  dash: [6,4] },
    { y: 65, label: I18n.t('chart.who.high'), color: 'rgba(198,40,40,.5)', dash: [4,4] },
  ];
}

/* ── Format helpers ─────────────────────────────────────────────────────── */
function formatDate(iso) {
  const [, m, d] = iso.split('-');
  const months = I18n.t('months');
  return `${parseInt(d)} ${months[parseInt(m)]}`;
}


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
          filter: item => item.datasetIndex === 0,
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              const level = val < 55 ? I18n.t('noise.low') : val < 65 ? I18n.t('noise.mid') : I18n.t('noise.high');
              return ` ${ctx.dataset.label}: ${val.toFixed(1)} dB — ${level}`;
            },
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
            text: I18n.t('chart.y.title'),
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
function updateChart(station, period) {
  if (!_chart) initChart();

  const forecastKey = period === 'daytime' ? 'daytime_forecast' : 'nighttime_forecast';
  const forecast    = station[forecastKey] || [];

  const labels  = forecast.map(e => formatDate(e.date));
  const values  = forecast.map(e => e.laeq);
  const pal     = PALETTE[period];
  const periodLabel = period === 'daytime' ? I18n.t('chart.period.daytime') : I18n.t('chart.period.nighttime');

  // Build WHO horizontal reference datasets (one point per day so they span full x-axis)
  const whoDatasets = getWhoLines().map(ref => ({
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
      label: `${I18n.t('chart.prediction')} (${periodLabel})`,
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
  _chart.options.scales.y.title.text = I18n.t('chart.y.title');

  _chart.update('active');
}

/* ── Public API ─────────────────────────────────────────────────────────── */
window.NoiseCharts = { initChart, updateChart };
