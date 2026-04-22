'use strict';

/* ── Translations ─────────────────────────────────────────────────────────── */
const TRANSLATIONS = {
  es: {
    'page.title':                 'Madrid · Predicción de Contaminación Acústica',
    'skip.link':                  'Saltar al contenido principal',
    'header.brand.city':          'Ayuntamiento de Madrid',
    'header.brand.title':         'Predicción de Contaminación Acústica',
    'header.updated.label':       'Última actualización:',
    'controls.aria':              'Filtros del mapa',
    'controls.period.legend':     'Período',
    'controls.period.daytime':    'Diurno',
    'controls.period.nighttime':  'Nocturno',
    'controls.period.hint':       'Seleccione el período horario para mostrar en el mapa',
    'controls.day.label':         'Día de predicción:',
    'controls.day.slider.aria':   'Seleccionar día de predicción (1–14 días desde hoy)',
    'error.message':              '<strong>Error:</strong> No se pudieron cargar los datos de predicción. Por favor, inténtelo de nuevo más tarde.',
    'loading.aria':               'Cargando datos…',
    'loading.text':               'Cargando predicciones…',
    'map.heading':                'Mapa de Estaciones',
    'map.aria':                   'Mapa interactivo de estaciones de ruido de Madrid. Haga clic en un marcador para ver la predicción detallada de esa estación.',
    'legend.title':               'Nivel LAeq (dB)',
    'legend.low':                 '< 55 dB — Bajo',
    'legend.mid':                 '55–65 dB — Moderado',
    'legend.high':                '≥ 65 dB — Alto',
    'legend.note':                'Umbrales basados en las directrices de la OMS para ruido ambiental (2018).',
    'popup.date':                 'Fecha',
    'popup.period':               'Período',
    'popup.daytime':              'Diurno (07–23 h)',
    'popup.nighttime':            'Nocturno (23–07 h)',
    'noise.low':                  'Bajo',
    'noise.mid':                  'Moderado',
    'noise.high':                 'Alto',
    'chart.heading.prefix':       'Predicción',
    'chart.station.placeholder':  'Seleccione una estación',
    'chart.hint':                 'Haga clic en un marcador del mapa para ver la predicción de 14 días.',
    'chart.aria':                 'Gráfico de líneas con la predicción de ruido a 14 días para la estación seleccionada',
    'chart.y.title':              'Nivel LAeq (dB)',
    'chart.who.mid':              'OMS: Moderado (55 dB)',
    'chart.who.high':             'OMS: Alto (65 dB)',
    'chart.period.daytime':       'Diurno',
    'chart.period.nighttime':     'Nocturno',
    'chart.prediction':           'Predicción',
    'stats.cluster.day':          'Clúster (diurno)',
    'stats.cluster.night':        'Clúster (nocturno)',
    'stats.max':                  'Máx. predicho (14 días)',
    'stats.min':                  'Mín. predicho (14 días)',
    'cluster.high':               'Ruido Alto',
    'cluster.low':                'Ruido Bajo',
    'cluster.mid':                'Ruido Medio',
    'cluster.n':                  'Clúster',
    'table.heading':              'Estaciones — Predicción de Ruido',
    'table.scroll.aria':          'Tabla de estaciones con ruido predicho',
    'table.caption':              'Estaciones ordenadas por nivel de ruido LAeq predicho el día seleccionado',
    'table.col.station':          'Estación',
    'table.col.laeq':             'LAeq predicho (dB)',
    'table.col.level':            'Nivel',
    'table.page.prev':            'Página anterior',
    'table.page.next':            'Página siguiente',
    'footer.data':                'Datos de ruido: <a href="https://datos.madrid.es" target="_blank" rel="noopener noreferrer">Portal de Datos Abiertos del Ayuntamiento de Madrid</a>. Predicciones generadas mediante modelos de aprendizaje automático y suavizados clásicos.',
    'footer.auto':                'Actualización automática diaria',
    'footer.updated':             'Actualizado:',
    'lang.toggle.label':          'EN',
    'lang.btn.aria':              'Switch language to English',
    'date.locale':                'es-ES',
    'months':                     ['','ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'],
  },
  en: {
    'page.title':                 'Madrid · Acoustic Noise Prediction',
    'skip.link':                  'Skip to main content',
    'header.brand.city':          'Madrid City Council',
    'header.brand.title':         'Acoustic Noise Prediction',
    'header.updated.label':       'Last updated:',
    'controls.aria':              'Map filters',
    'controls.period.legend':     'Period',
    'controls.period.daytime':    'Daytime',
    'controls.period.nighttime':  'Nighttime',
    'controls.period.hint':       'Select the time period to display on the map',
    'controls.day.label':         'Forecast day:',
    'controls.day.slider.aria':   'Select forecast day (1–14 days from today)',
    'error.message':              '<strong>Error:</strong> Could not load forecast data. Please try again later.',
    'loading.aria':               'Loading data…',
    'loading.text':               'Loading forecasts…',
    'map.heading':                'Station Map',
    'map.aria':                   'Interactive map of Madrid noise monitoring stations. Click a marker to see the detailed forecast for that station.',
    'legend.title':               'LAeq Level (dB)',
    'legend.low':                 '< 55 dB — Low',
    'legend.mid':                 '55–65 dB — Moderate',
    'legend.high':                '≥ 65 dB — High',
    'legend.note':                'Thresholds based on WHO Environmental Noise Guidelines (2018).',
    'popup.date':                 'Date',
    'popup.period':               'Period',
    'popup.daytime':              'Daytime (07–23 h)',
    'popup.nighttime':            'Nighttime (23–07 h)',
    'noise.low':                  'Low',
    'noise.mid':                  'Moderate',
    'noise.high':                 'High',
    'chart.heading.prefix':       'Forecast',
    'chart.station.placeholder':  'Select a station',
    'chart.hint':                 'Click a map marker to see the 14-day forecast.',
    'chart.aria':                 'Line chart with 14-day noise forecast for the selected station',
    'chart.y.title':              'LAeq Level (dB)',
    'chart.who.mid':              'WHO: Moderate (55 dB)',
    'chart.who.high':             'WHO: High (65 dB)',
    'chart.period.daytime':       'Daytime',
    'chart.period.nighttime':     'Nighttime',
    'chart.prediction':           'Forecast',
    'stats.cluster.day':          'Cluster (daytime)',
    'stats.cluster.night':        'Cluster (nighttime)',
    'stats.max':                  'Max. forecast (14 days)',
    'stats.min':                  'Min. forecast (14 days)',
    'cluster.high':               'High Noise',
    'cluster.low':                'Low Noise',
    'cluster.mid':                'Medium Noise',
    'cluster.n':                  'Cluster',
    'table.heading':              'Stations — Noise Forecast',
    'table.scroll.aria':          'Table of stations with predicted noise levels',
    'table.caption':              'Stations sorted by predicted LAeq noise level for the selected day',
    'table.col.station':          'Station',
    'table.col.laeq':             'Predicted LAeq (dB)',
    'table.col.level':            'Level',
    'table.page.prev':            'Previous page',
    'table.page.next':            'Next page',
    'footer.data':                'Noise data: <a href="https://datos.madrid.es" target="_blank" rel="noopener noreferrer">Madrid Open Data Portal</a>. Forecasts generated using machine learning models and classical smoothing methods.',
    'footer.auto':                'Automatic daily update',
    'footer.updated':             'Updated:',
    'lang.toggle.label':          'ES',
    'lang.btn.aria':              'Cambiar idioma a español',
    'date.locale':                'en-GB',
    'months':                     ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
  },
};

/* ── Module state ─────────────────────────────────────────────────────────── */
let _lang = localStorage.getItem('dashboard-lang') || 'es';

/* ── Core API ─────────────────────────────────────────────────────────────── */
function t(key) {
  return (TRANSLATIONS[_lang] && TRANSLATIONS[_lang][key]) ||
         (TRANSLATIONS.es[key]) ||
         key;
}

function getLang() { return _lang; }

function setLang(lang) {
  if (!TRANSLATIONS[lang]) return;
  _lang = lang;
  localStorage.setItem('dashboard-lang', lang);
  applyAll();
  document.dispatchEvent(new CustomEvent('langchange', { detail: { lang } }));
}

/* ── DOM application ──────────────────────────────────────────────────────── */
function applyAll() {
  document.documentElement.lang = _lang;
  document.title = t('page.title');

  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });

  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    el.innerHTML = t(el.dataset.i18nHtml);
  });

  document.querySelectorAll('[data-i18n-aria]').forEach(el => {
    el.setAttribute('aria-label', t(el.dataset.i18nAria));
  });

  const btn = document.getElementById('lang-toggle');
  if (btn) {
    btn.textContent = t('lang.toggle.label');
    btn.setAttribute('aria-label', t('lang.btn.aria'));
    btn.setAttribute('lang', _lang === 'es' ? 'en' : 'es');
  }
}

/* ── Init ─────────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  applyAll();

  const btn = document.getElementById('lang-toggle');
  if (btn) {
    btn.addEventListener('click', () => setLang(_lang === 'es' ? 'en' : 'es'));
  }
});

window.I18n = { t, getLang, setLang, applyAll };
