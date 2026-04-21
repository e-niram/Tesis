"""
05_update_predictions.py — Fit Exponential Smoothing per station and write
app/data/predictions.json for the static frontend.

Called daily by GitHub Actions immediately after 04_fetch_api.py.

For each station, the best ES model structure (trend type, seasonality) is
loaded from the cluster-level config selected by 06_train_and_save_models.py.
ES is then re-fitted on the station's own history, so forecasts are fully
individualised while the model structure is informed by the cluster.

Output
------
app/data/predictions.json
  {
    "last_updated": "YYYY-MM-DD",
    "forecast_horizon_days": 14,
    "stations": [
      {
        "id": 1,
        "name": "Paseo de Recoletos",
        "lat": 40.422599,
        "lon": -3.691877,
        "cluster_day": 0,
        "cluster_night": 0,
        "daytime_forecast":  [{"date": "YYYY-MM-DD", "laeq": 67.4}, …],
        "nighttime_forecast": [{"date": "YYYY-MM-DD", "laeq": 54.2}, …]
      }, …
    ]
  }
"""

import warnings
from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
CLUSTER_DIR  = REPO_ROOT / "results" / "clustering" / "metrics"
FINAL_DIR    = REPO_ROOT / "data" / "final"
APP_DATA_DIR = REPO_ROOT / "app" / "data"

ASSIGNMENT_CSVS = {
    "daytime":   CLUSTER_DIR / "Results_daytime_k3_s48.csv",
    "nighttime": CLUSTER_DIR / "Results_nighttime_k3_s47.csv",
}

FINAL_CSVS = {
    "daytime":   FINAL_DIR / "daytime_final.csv",
    "nighttime": FINAL_DIR / "nighttime_final.csv",
}

LEAD    = 14
MIN_OBS = 14  # minimum observations needed for weekly-seasonal ES

# Fixed model structure for all stations — Holt-Winters Additive with weekly cycle
ES_KWARGS = dict(trend="add", damped_trend=False, seasonal="add", seasonal_periods=7)

# ── Station metadata ───────────────────────────────────────────────────────────
STATIONS = {
    1: "Paseo de Recoletos",
    2: "Carlos V",
    3: "Plaza del Carmen",
    5: "Barrio del Pilar",
    6: "Gregorio Marañón",
    8: "Escuelas Aguirre",
    10: "Cuatro Caminos",
    11: "Ramón y Cajal",
    12: "Manuel Becerra",
    13: "Vallecas",
    14: "Plaza Elíptica",
    16: "Arturo Soria",
    17: "Villaverde",
    18: "Farolillo",
    19: "Alto Extremadura",
    20: "Avenida de Moratalaz",
    24: "Casa de Campo",
    25: "Santa Eugenia",
    26: "Embajada",
    27: "Barajas Pueblo",
    28: "Cuatro Vientos",
    29: "El Pardo",
    30: "Campo de las Naciones",
    31: "Sanchinarro",
    47: "Mendez Álvaro",
    48: "Castellana",
    50: "Plaza de Castilla",
    54: "Ensanche de Vallecas",
    55: "Urb Embajada 2",
    86: "Tres Olivos",
}

COORDINATES = {
    1:  [40.422599, -3.691877],
    2:  [40.409121, -3.691509],
    3:  [40.419251, -3.703175],
    5:  [40.478197, -3.711543],
    6:  [40.437548, -3.690758],
    8:  [40.421575, -3.682377],
    10: [40.445581, -3.707157],
    11: [40.451522, -3.677359],
    12: [40.428741, -3.668584],
    13: [40.388116, -3.651506],
    14: [40.384978, -3.718807],
    16: [40.440013, -3.639284],
    17: [40.347100, -3.713312],
    18: [40.394788, -3.731786],
    19: [40.407871, -3.741950],
    20: [40.408028, -3.645215],
    24: [40.419390, -3.747319],
    25: [40.379072, -3.602561],
    26: [40.459331, -3.580128],
    27: [40.476929, -3.580101],
    28: [40.375250, -3.777830],
    29: [40.517989, -3.774551],
    30: [40.460761, -3.616517],
    31: [40.494254, -3.660454],
    47: [40.398022, -3.686808],
    48: [40.439854, -3.690302],
    50: [40.465628, -3.688719],
    54: [40.372993, -3.612100],
    55: [40.462298, -3.580599],
    86: [40.500598, -3.689643],
}


# ── Per-station forecasting ────────────────────────────────────────────────────

def forecast_station(series: np.ndarray, last_date: date) -> list[dict]:
    """
    Fit Holt-Winters Additive on this station's history and return a 14-day
    forecast as a list of {"date", "laeq"} dicts.
    Falls back to level-only ES if there are too few observations.
    """
    clean = series[~np.isnan(series)]
    kwargs = ES_KWARGS if len(clean) >= MIN_OBS else dict(
        trend=None, damped_trend=False, seasonal=None, seasonal_periods=None
    )

    fallback = float(np.nanmean(clean)) if len(clean) else 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(clean, **kwargs).fit(optimized=True, disp=False)
        preds = fit.forecast(LEAD)
    except Exception:
        preds = np.full(LEAD, fallback)

    preds = np.where(np.isnan(preds), fallback, preds)

    return [
        {"date": str(last_date + timedelta(days=h + 1)), "laeq": round(float(v), 2)}
        for h, v in enumerate(preds)
    ]


# ── Cluster assignment loader ──────────────────────────────────────────────────

def load_assignments(period: str) -> dict[int, int]:
    df = pd.read_csv(ASSIGNMENT_CSVS[period], sep=";")
    return dict(zip(df["Station_ID"].astype(int), df["Cluster"].astype(int)))


# ── Build predictions.json ─────────────────────────────────────────────────────

def build_predictions_json() -> dict:
    today = date.today()

    # Load per-station time series for both periods
    series_store: dict[str, tuple[pd.DataFrame, date]] = {}
    for period in ("daytime", "nighttime"):
        df = pd.read_csv(FINAL_CSVS[period], sep=";", index_col=0)
        df.index = pd.to_datetime(df.index)
        df.columns = df.columns.astype(str)
        last_date = df.index[-1].date()
        series_store[period] = (df, last_date)

    # Load cluster assignments
    assignments = {p: load_assignments(p) for p in ("daytime", "nighttime")}

    # Build per-station forecasts
    station_list = []
    for sid, name in STATIONS.items():
        lat, lon = COORDINATES[sid]
        cday   = assignments["daytime"].get(sid, 0)
        cnight = assignments["nighttime"].get(sid, 0)

        row: dict = {
            "id": sid, "name": name, "lat": lat, "lon": lon,
            "cluster_day": cday, "cluster_night": cnight,
        }

        for period, key in [("daytime", "daytime_forecast"), ("nighttime", "nighttime_forecast")]:
            df, last_date = series_store[period]
            col = str(sid)
            series = df[col].to_numpy(dtype=float) if col in df.columns else np.array([])

            print(f"  {name:<30} {period} …")
            row[key] = forecast_station(series, last_date)

        station_list.append(row)

    return {
        "last_updated": str(today),
        "forecast_horizon_days": LEAD,
        "stations": station_list,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Building predictions.json …\n")
    payload = build_predictions_json()

    out_path = APP_DATA_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"\nDone.  {len(payload['stations'])} stations → {out_path}")
