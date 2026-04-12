"""
05_update_predictions.py — Load pre-trained RF models, extend cluster means
with the newly appended day, forecast the next 14 days, and write
app/data/predictions.json for the static frontend.

Called daily by GitHub Actions immediately after 04_fetch_api.py.

No model retraining occurs here.  The models in models/ were trained by
06_train_and_save_models.py and are reloaded via joblib for inference only.

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

import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parents[2]
MODEL_DIR    = REPO_ROOT / "models"
CLUSTER_DIR  = REPO_ROOT / "results" / "clustering" / "metrics"
FINAL_DIR    = REPO_ROOT / "data" / "final"
APP_DATA_DIR = REPO_ROOT / "app" / "data"

CLUSTER_CSVS = {
    "daytime":   CLUSTER_DIR / "Cluster_Means_daytime.csv",
    "nighttime": CLUSTER_DIR / "Cluster_Means_nighttime.csv",
}

ASSIGNMENT_CSVS = {
    "daytime":   CLUSTER_DIR / "Results_daytime_k3_s48.csv",
    "nighttime": CLUSTER_DIR / "Results_nighttime_k3_s47.csv",
}

FINAL_CSVS = {
    "daytime":   FINAL_DIR / "daytime_final.csv",
    "nighttime": FINAL_DIR / "nighttime_final.csv",
}

LEAD = 14

# ── Station metadata (from src/constants.py) ───────────────────────────────────
STATIONS = {
    1: "Paseo de Recoletos",
    2: "Carlos V",
    3: "Plaza del Carmen",
    4: "Plaza de España",
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
    4:  [40.424005, -3.712253],
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
    30: [40.460761, -3.616317],
    31: [40.494254, -3.660454],
    47: [40.398022, -3.686808],
    48: [40.439854, -3.690302],
    50: [40.465628, -3.688719],
    54: [40.372993, -3.612100],
    55: [40.462298, -3.580599],
    86: [40.500598, -3.689643],
}


# ── Physical conversion ────────────────────────────────────────────────────────

def db_to_pressure(db: np.ndarray) -> np.ndarray:
    return 10 ** (db / 20)


def pressure_to_db(p: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.where(p > 0, p, 1e-12))


# ── Cluster mean for a new day ─────────────────────────────────────────────────

def compute_cluster_means_for_date(
    period: str,
    target_date: date,
) -> dict[str, float]:
    """
    Compute the energy-average dB value for each cluster on *target_date*
    using the newly appended data in data/final/.

    Returns {cluster_col: dB_value}, e.g. {"Cluster_0": 67.4, "Cluster_1": …}
    """
    final_df = pd.read_csv(FINAL_CSVS[period], sep=";", index_col=0)
    final_df.index = pd.to_datetime(final_df.index)

    date_str = pd.Timestamp(target_date)
    if date_str not in final_df.index:
        raise KeyError(f"Date {target_date} not found in {FINAL_CSVS[period]}")

    row = final_df.loc[date_str]

    assign = pd.read_csv(ASSIGNMENT_CSVS[period], sep=";")
    assign["Station_ID"] = assign["Station_ID"].astype(str)

    cluster_means: dict[str, float] = {}
    for cluster_id, grp in assign.groupby("Cluster"):
        station_ids = [str(s) for s in grp["Station_ID"].tolist()]
        vals = row[station_ids].astype(float).dropna().to_numpy()
        if len(vals) == 0:
            cluster_means[f"Cluster_{cluster_id}"] = float("nan")
        else:
            cluster_means[f"Cluster_{cluster_id}"] = float(
                pressure_to_db(db_to_pressure(vals).mean())
            )
    return cluster_means


def append_cluster_means(period: str, target_date: date) -> None:
    """Append a new row to Cluster_Means_{period}.csv if not already present."""
    csv_path = CLUSTER_CSVS[period]
    df = pd.read_csv(csv_path, sep=";", parse_dates=["FECHA"])
    date_ts = pd.Timestamp(target_date)

    if date_ts in df["FECHA"].values:
        print(f"  Cluster_Means_{period}: {target_date} already present.")
        return

    new_means = compute_cluster_means_for_date(period, target_date)
    new_row = {"FECHA": str(target_date), **new_means}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, sep=";", index=False)
    print(f"  Cluster_Means_{period}: appended {target_date}.")


# ── Feature engineering (mirrors ml_predictions.py / 06_train_and_save.py) ────

LAGS: list[int] = list(range(1, 15)) + [21, 28, 35, 42, 365]


def _cyclic_encode(arr: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * arr / period
    return np.sin(angle), np.cos(angle)


def build_features(series: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    n = len(series)
    cols: list[np.ndarray] = []
    for k in LAGS:
        col = np.full(n, np.nan)
        col[k:] = series[: n - k]
        cols.append(col)
    sin_doy,   cos_doy   = _cyclic_encode(dates.dayofyear.to_numpy(), 365.25)
    sin_dow,   cos_dow   = _cyclic_encode(dates.dayofweek.to_numpy(), 7.0)
    sin_month, cos_month = _cyclic_encode(dates.month.to_numpy(),     12.0)
    cols += [sin_doy, cos_doy, sin_dow, cos_dow, sin_month, cos_month]
    return np.column_stack(cols)


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_cluster(period: str, cluster_col: str) -> list[dict]:
    """
    Load the saved model bundle for (period, cluster) and produce a 14-day
    forecast starting from the day after the last observation in the
    Cluster_Means CSV.

    Returns list of {"date": "YYYY-MM-DD", "laeq": float}.
    """
    model_path = MODEL_DIR / f"rf_{period}_{cluster_col.lower()}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run src/pipeline/06_train_and_save_models.py first."
        )

    bundle = joblib.load(model_path)
    model    = bundle["model"]
    scaler_X = bundle["scaler_X"]
    scaler_y = bundle["scaler_y"]

    df = pd.read_csv(CLUSTER_CSVS[period], sep=";", parse_dates=["FECHA"])
    dates  = pd.DatetimeIndex(df["FECHA"])
    series = df[cluster_col].to_numpy(dtype=float)

    # Build feature matrix and take the very last row for inference
    X_all = build_features(series, dates)
    x_last = X_all[[-1]]  # shape (1, n_features)

    x_scaled = scaler_X.transform(x_last)
    y_hat_sc = model.predict(x_scaled)                        # (1, LEAD)
    y_hat    = scaler_y.inverse_transform(y_hat_sc).ravel()   # (LEAD,)

    last_date = dates[-1].date()
    return [
        {"date": str(last_date + timedelta(days=h + 1)), "laeq": round(float(v), 2)}
        for h, v in enumerate(y_hat)
    ]


# ── Load cluster assignments ───────────────────────────────────────────────────

def load_assignments(period: str) -> dict[int, int]:
    """Returns {station_id: cluster_id}."""
    df = pd.read_csv(ASSIGNMENT_CSVS[period], sep=";")
    return dict(zip(df["Station_ID"].astype(int), df["Cluster"].astype(int)))


# ── Build predictions.json ─────────────────────────────────────────────────────

def build_predictions_json() -> dict:
    today = date.today()

    # ── Step 1: update Cluster_Means CSVs with yesterday's data ──────────────
    yesterday = today - timedelta(days=1)
    for period in ("daytime", "nighttime"):
        try:
            append_cluster_means(period, yesterday)
        except KeyError as exc:
            print(f"  WARNING: {exc} — skipping cluster mean update for {period}.")

    # ── Step 2: per-period, per-cluster forecasts ─────────────────────────────
    forecasts: dict[str, dict[str, list[dict]]] = {}

    for period in ("daytime", "nighttime"):
        forecasts[period] = {}
        df = pd.read_csv(CLUSTER_CSVS[period], sep=";")
        cluster_cols = [c for c in df.columns if c.upper().startswith("CLUSTER_")]

        for col in cluster_cols:
            print(f"  Predicting {period} / {col} …")
            try:
                forecasts[period][col] = predict_cluster(period, col)
            except FileNotFoundError as exc:
                print(f"  ERROR: {exc}")
                forecasts[period][col] = []

    # ── Step 3: map cluster forecasts → individual stations ───────────────────
    day_assign   = load_assignments("daytime")
    night_assign = load_assignments("nighttime")

    station_list = []
    for sid, name in STATIONS.items():
        lat, lon = COORDINATES[sid]
        cday   = day_assign.get(sid, 0)
        cnight = night_assign.get(sid, 0)

        station_list.append({
            "id":    sid,
            "name":  name,
            "lat":   lat,
            "lon":   lon,
            "cluster_day":   cday,
            "cluster_night": cnight,
            "daytime_forecast":  forecasts["daytime"].get(f"Cluster_{cday}",  []),
            "nighttime_forecast": forecasts["nighttime"].get(f"Cluster_{cnight}", []),
        })

    return {
        "last_updated": str(today),
        "forecast_horizon_days": LEAD,
        "stations": station_list,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Building predictions.json …")
    payload = build_predictions_json()

    out_path = APP_DATA_DIR / "predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_stations = len(payload["stations"])
    print(f"Done.  {n_stations} stations written to {out_path}")
