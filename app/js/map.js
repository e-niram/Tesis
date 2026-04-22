/**
 * map.js — Leaflet interactive map for the Madrid Noise Forecast dashboard.
 *
 * Exports:
 *   initMap(onStationClick)  → leafletMap instance
 *   updateMapMarkers(stations, period, dayIndex)
 */

'use strict';

/* ── Colour helpers ─────────────────────────────────────────────────────── */

/**
 * WHO Environmental Noise Guidelines (2018) thresholds.
 * Returns a CSS colour string and a badge label.
 */
function noiseLevel(laeq) {
  if (laeq < 55) return { color: '#2e7d32', label: I18n.t('noise.low'),  cssClass: 'low'  };
  if (laeq < 65) return { color: '#d97706', label: I18n.t('noise.mid'),  cssClass: 'mid'  };
  return              { color: '#c62828', label: I18n.t('noise.high'), cssClass: 'high' };
}

/**
 * Create a circular SVG marker with a contrasting dB label inside.
 * Using DivIcon keeps us dependency-free while providing full styling control.
 */
function makeMarkerIcon(laeq, isSelected) {
  const { color } = noiseLevel(laeq);
  const size   = isSelected ? 42 : 34;
  const border = isSelected ? `3px solid #FFD700` : `2px solid rgba(255,255,255,.7)`;
  const html = `
    <div style="
      width:${size}px; height:${size}px;
      background:${color};
      border-radius:50%;
      border:${border};
      display:flex; align-items:center; justify-content:center;
      box-shadow: 0 2px 6px rgba(0,0,0,.35);
      font-family: 'Source Sans Pro', sans-serif;
      font-size:${isSelected ? 11 : 10}px;
      font-weight:700;
      color:#fff;
      line-height:1;
      cursor:pointer;
    ">${laeq.toFixed(1)}</div>`;
  return L.divIcon({
    html,
    className: '',
    iconSize:   [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor:[0, -(size / 2 + 4)],
  });
}

/* ── Module state ───────────────────────────────────────────────────────── */
let _map           = null;
let _markers       = {};       // { stationId: L.marker }
let _selectedId    = null;
let _onStationClick = null;

/* ── Init ───────────────────────────────────────────────────────────────── */

/**
 * Initialise and return the Leaflet map instance.
 * @param {function} onStationClick  Called with (stationData) when a marker is clicked.
 */
function initMap(onStationClick) {
  _onStationClick = onStationClick;

  _map = L.map('map', {
    center: [40.42, -3.70],
    zoom:   11,
    zoomControl: true,
    attributionControl: true,
  });

  // Carto Positron tile layer — clean, light, free
  L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> &copy; ' +
        '<a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 19,
    }
  ).addTo(_map);

  return _map;
}

/* ── Update markers ─────────────────────────────────────────────────────── */

/**
 * Place / refresh one coloured circle marker per station.
 *
 * @param {Array}  stations   Full station list from predictions.json
 * @param {string} period     'daytime' | 'nighttime'
 * @param {number} dayIndex   0-based day index into the 14-day forecast
 */
function updateMapMarkers(stations, period, dayIndex) {
  const forecastKey = period === 'daytime' ? 'daytime_forecast' : 'nighttime_forecast';

  stations.forEach(station => {
    const forecast = station[forecastKey];
    if (!forecast || forecast.length === 0) return;

    const entry  = forecast[Math.min(dayIndex, forecast.length - 1)];
    const laeq   = entry.laeq;
    const date   = entry.date;
    const isSelected = station.id === _selectedId;
    const icon   = makeMarkerIcon(laeq, isSelected);
    const { label } = noiseLevel(laeq);

    if (_markers[station.id]) {
      // Update existing marker
      _markers[station.id].setIcon(icon);
      _markers[station.id].setPopupContent(_buildPopup(station, laeq, date, label, period));
    } else {
      // Create new marker
      const marker = L.marker([station.lat, station.lon], {
        icon,
        title: station.name,   // tooltip for keyboard users
        alt:   `${station.name}: ${laeq.toFixed(1)} dB`,
        keyboard: true,
        riseOnHover: true,
      });

      marker.bindPopup(_buildPopup(station, laeq, date, label, period), {
        maxWidth: 240,
        className: 'noise-popup',
      });

      marker.on('click keypress', (e) => {
        if (e.type === 'keypress' && e.originalEvent.key !== 'Enter') return;
        _selectedId = station.id;
        _onStationClick(station);
        // Refresh all marker icons to reflect new selection
        updateMapMarkers(stations, period, dayIndex);
      });

      marker.addTo(_map);
      _markers[station.id] = marker;
    }
  });
}

function _buildPopup(station, laeq, date, label, period) {
  const periodLabel = period === 'daytime' ? I18n.t('popup.daytime') : I18n.t('popup.nighttime');
  const { color }   = noiseLevel(laeq);
  return `
    <div class="popup-title">${_escapeHtml(station.name)}</div>
    <div class="popup-laeq-row">
      <span class="popup-db">${laeq.toFixed(1)} dB</span>
      <span class="popup-level-badge" style="background:${color}">${_escapeHtml(label)}</span>
    </div>
    <dl class="popup-meta">
      <dt>${I18n.t('popup.date')}</dt><dd>${_escapeHtml(date)}</dd>
      <dt>${I18n.t('popup.period')}</dt><dd>${periodLabel}</dd>
    </dl>`;
}

function _escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── Public API ─────────────────────────────────────────────────────────── */
window.NoiseMap = { initMap, updateMapMarkers, noiseLevel };
