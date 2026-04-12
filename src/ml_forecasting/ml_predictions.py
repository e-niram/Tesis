"""
ml_predictions.py — Rolling-window ML forecast for cluster noise levels.

Mirrors predictions.sas (PROC ESM rolling forecast):
  • Expanding window starting at 3 500 training observations
  • Step / lead   : 14 days
  • Metrics       : MAE, MSE, RMSE, MAPE  (per window)

Models
------
  random_forest : RandomForestRegressor  (sklearn)
  mlp           : MLPRegressor           (sklearn)

Strategy: direct multi-output regression.
  X[i] = lag features + cyclic calendar features at day i
  y[i] = [series[i+1], …, series[i+14]]  (14 simultaneous targets)

This avoids the compounding errors of a recursive 1-step-ahead approach and
lets each model learn a different mapping for each forecast horizon.

Outputs  →  results/ml_forecasting/
  rolling_forecasts_<period>_<cluster>_<model>.csv   per-observation predictions
  forecast_metrics_<period>_<cluster>_<model>.csv    per-window MAE/MSE/RMSE/MAPE
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "results" / "clustering" / "metrics"
OUT_DIR   = REPO_ROOT / "results" / "ml_forecasting"

# ── Forecast parameters ────────────────────────────────────────────────────────
TRAIN_OBS = 3_500   # initial training window size (days) — mirrors SAS train_obs
LEAD      = 14      # forecast horizon (days)          — mirrors SAS lead

# Lag indices (days back from current observation).
# Covers: daily (1–14), weekly multiples (21, 28, 35, 42), and annual (365).
LAGS: list[int] = list(range(1, 15)) + [21, 28, 35, 42, 365]

# ── Model definitions ──────────────────────────────────────────────────────────
MODELS: dict = {
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    # MLPRegressor with scaled targets (see rolling_forecast).
    # Higher lr + scaled y converges in ~30 iterations instead of 500+.
    "mlp": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-2,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=42,
    ),
}


# ── Feature engineering ────────────────────────────────────────────────────────

def _cyclic_encode(arr: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Sine / cosine encoding for a cyclic variable (e.g. day-of-year)."""
    angle = 2.0 * np.pi * arr / period
    return np.sin(angle), np.cos(angle)


def build_features(series: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Build a feature matrix of shape (n, n_features).

    Rows 0 … max(LAGS)-1 contain NaN lag values; they are never used for
    training or prediction because build_supervised / rolling_forecast skip
    them.

    Features
    --------
    lag_k         : series[i - k]  for each k in LAGS
    sin/cos_doy   : cyclic day-of-year   (period 365.25)
    sin/cos_dow   : cyclic day-of-week   (period 7)
    sin/cos_month : cyclic month-of-year (period 12)
    """
    n    = len(series)
    cols: list[np.ndarray] = []

    for k in LAGS:
        col      = np.full(n, np.nan)
        col[k:]  = series[: n - k]
        cols.append(col)

    sin_doy,   cos_doy   = _cyclic_encode(dates.dayofyear.to_numpy(),  365.25)
    sin_dow,   cos_dow   = _cyclic_encode(dates.dayofweek.to_numpy(),  7.0)
    sin_month, cos_month = _cyclic_encode(dates.month.to_numpy(),      12.0)
    cols += [sin_doy, cos_doy, sin_dow, cos_dow, sin_month, cos_month]

    return np.column_stack(cols)


def build_supervised(
    series: np.ndarray,
    dates:  pd.DatetimeIndex,
    lead:   int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) for direct multi-output regression.

      X[i] : feature vector at position i  (lags + calendar)
      y[i] : series[i+1 : i+1+lead]        (next `lead` values)

    Only rows where every lag exists AND a full lead window is available
    are included.
    """
    max_lag = max(LAGS)
    X_all   = build_features(series, dates)

    # valid range: lags exist (i >= max_lag)  AND  targets fit (i + lead < n)
    valid = np.arange(max_lag, len(series) - lead)

    X = X_all[valid]
    y = np.array([series[i + 1 : i + 1 + lead] for i in valid])
    return X, y


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    err      = actual - forecast
    abs_err  = np.abs(err)
    sq_err   = err ** 2
    pct_err  = np.where(actual > 0, abs_err / actual * 100.0, np.nan)
    return {
        "MAE":  float(np.mean(abs_err)),
        "MSE":  float(np.mean(sq_err)),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "MAPE": float(np.nanmean(pct_err)),
    }


# ── Rolling forecast ───────────────────────────────────────────────────────────

def rolling_forecast(
    series:      np.ndarray,
    dates:       pd.DatetimeIndex,
    base_model,
    model_name:  str,
    cluster_id:  str,
    time_period: str,
    train_obs:   int = TRAIN_OBS,
    lead:        int = LEAD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window rolling forecast identical in structure to predictions.sas.

    Each iteration:
      1. Uses observations 1 … current_end as training data.
      2. Builds supervised (X, y) from the training slice.
      3. Fits a fresh model clone (scaler fitted on training data only).
      4. Forecasts the next `lead` days from the last training observation.
      5. Records per-observation predictions and per-window metrics.
      6. Advances current_end by `lead`.

    Returns
    -------
    all_forecasts : DataFrame  — one row per (window × horizon)
    metrics       : DataFrame  — one row per window (MAE, MSE, RMSE, MAPE)
    """
    n_total    = len(series)
    n_windows  = (n_total - train_obs) // lead
    print(
        f"  [{time_period} / {cluster_id} / {model_name}]  "
        f"n={n_total}  train_obs={train_obs}  lead={lead}  "
        f"expected_windows={n_windows}"
    )

    forecast_rows: list[dict] = []
    metrics_rows:  list[dict] = []

    current_end = train_obs
    window_num  = 1

    while current_end + lead <= n_total:
        tr_series = series[:current_end]
        tr_dates  = dates[:current_end]

        X_train, y_train = build_supervised(tr_series, tr_dates, lead)

        if len(X_train) == 0:
            print(f"    Window {window_num}: insufficient training samples — skipping.")
            current_end += lead
            window_num  += 1
            continue

        # Scale features on training data only (no data leakage)
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_train)

        # Scale targets — critical for MLP convergence speed; RF is invariant
        # to monotonic target transforms so this never hurts.
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_train)

        # Fresh model clone → fit
        model = clone(base_model)
        model.fit(X_scaled, y_scaled)

        # Predict from the last training observation, then inverse-scale
        X_all_train = build_features(tr_series, tr_dates)
        x_pred      = scaler_X.transform(X_all_train[[-1]])        # (1, n_features)
        y_hat_sc    = model.predict(x_pred)                        # (1, lead)
        y_hat       = scaler_y.inverse_transform(y_hat_sc).ravel() # (lead,)

        actuals      = series[current_end : current_end + lead]
        future_dates = dates[current_end : current_end + lead]

        # Per-observation rows (mirrors work.all_forecasts in SAS)
        for h, (dt, actual, pred) in enumerate(
            zip(future_dates, actuals, y_hat), start=1
        ):
            err     = float(actual) - float(pred)
            abs_err = abs(err)
            forecast_rows.append({
                "window_num":    window_num,
                "train_end_obs": current_end,
                "horizon":       h,
                "fecha":         dt.date().isoformat(),
                "actual":        round(float(actual), 6),
                "forecast_val":  round(float(pred),   6),
                "error":         round(err,            6),
                "abs_error":     round(abs_err,        6),
                "sq_error":      round(err ** 2,       6),
                "pct_error":     round(abs_err / float(actual) * 100, 6)
                                 if float(actual) > 0 else None,
            })

        # Per-window summary (mirrors work.forecast_metrics in SAS)
        m = compute_metrics(actuals, y_hat)
        metrics_rows.append({
            "window_num":     window_num,
            "train_end_obs":  current_end,
            "forecast_start": future_dates[0].date().isoformat(),
            "forecast_end":   future_dates[-1].date().isoformat(),
            "n":              lead,
            "MAE":            round(m["MAE"],  4),
            "MSE":            round(m["MSE"],  4),
            "RMSE":           round(m["RMSE"], 4),
            "MAPE":           round(m["MAPE"], 4),
        })

        print(
            f"    Window {window_num:3d} | train 1–{current_end} "
            f"| {future_dates[0].date()} → {future_dates[-1].date()} "
            f"| MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.4f}"
        )

        current_end += lead
        window_num  += 1

    return pd.DataFrame(forecast_rows), pd.DataFrame(metrics_rows)


# ── Per-period runner ──────────────────────────────────────────────────────────

def run_period(period: str, csv_path: Path) -> None:
    """Load one Cluster_Means CSV and run all cluster × model combinations."""
    df      = pd.read_csv(csv_path, sep=";", parse_dates=["FECHA"])
    dates   = pd.DatetimeIndex(df["FECHA"])
    cluster_cols = [c for c in df.columns if c.upper().startswith("CLUSTER_")]

    for cluster_col in cluster_cols:
        cluster_id = cluster_col.lower()          # e.g. "cluster_0"
        series     = df[cluster_col].to_numpy(dtype=float)

        for model_name, base_model in MODELS.items():
            print()
            all_fcst, metrics = rolling_forecast(
                series,
                dates,
                base_model  = base_model,
                model_name  = model_name,
                cluster_id  = cluster_id,
                time_period = period,
            )

            stem = f"{period}_{cluster_id}_{model_name}"
            all_fcst.to_csv(OUT_DIR / f"rolling_forecasts_{stem}.csv", index=False)
            metrics.to_csv( OUT_DIR / f"forecast_metrics_{stem}.csv",  index=False)
            print(f"    → Saved results for {stem}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_period("daytime",   DATA_DIR / "Cluster_Means_daytime.csv")
    run_period("nighttime", DATA_DIR / "Cluster_Means_nighttime.csv")

    print(f"\nDone.  All results saved to: {OUT_DIR}")
