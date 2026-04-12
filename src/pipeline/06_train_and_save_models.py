"""
06_train_and_save_models.py — Train RandomForest models on the full dataset and save with joblib.

Run this once (or monthly via GitHub Actions) to produce the pre-trained models
that 05_update_predictions.py loads for daily inference.

No rolling windows here — we fit on ALL available data so the model sees the
most recent observations before being frozen for daily forecasting.

Output:  models/rf_{period}_cluster_{i}.pkl   (6 files: 2 periods × 3 clusters)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "results" / "clustering" / "metrics"
MODEL_DIR = REPO_ROOT / "models"

# ── Reuse constants from ml_predictions ───────────────────────────────────────
LAGS: list[int] = list(range(1, 15)) + [21, 28, 35, 42, 365]
LEAD = 14

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)


# ── Feature engineering (mirrors ml_predictions.py) ──────────────────────────

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


def build_supervised(
    series: np.ndarray,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct multi-output supervised dataset (X → next LEAD values)."""
    max_lag = max(LAGS)
    X_all = build_features(series, dates)
    valid = np.arange(max_lag, len(series) - LEAD)
    X = X_all[valid]
    y = np.array([series[i + 1 : i + 1 + LEAD] for i in valid])
    return X, y


# ── Training ───────────────────────────────────────────────────────────────────

def train_and_save(period: str, csv_path: Path) -> None:
    """Fit one RF per cluster and persist model + scalers to disk."""
    df = pd.read_csv(csv_path, sep=";", parse_dates=["FECHA"])
    dates = pd.DatetimeIndex(df["FECHA"])
    cluster_cols = [c for c in df.columns if c.upper().startswith("CLUSTER_")]

    for cluster_col in cluster_cols:
        cluster_id = cluster_col.lower()  # e.g. "cluster_0"
        series = df[cluster_col].to_numpy(dtype=float)

        print(f"\n  Training  {period} / {cluster_id}  (n={len(series)}) …")
        X, y = build_supervised(series, dates)

        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_scaled, y_scaled)

        bundle = {
            "model":    model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "period":   period,
            "cluster":  cluster_id,
            "trained_on_obs": len(series),
            "lead":     LEAD,
            "lags":     LAGS,
        }

        out_path = MODEL_DIR / f"rf_{period}_{cluster_id}.pkl"
        joblib.dump(bundle, out_path, compress=3)
        print(f"  → Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_and_save("daytime",   DATA_DIR / "Cluster_Means_daytime.csv")
    train_and_save("nighttime", DATA_DIR / "Cluster_Means_nighttime.csv")

    print(f"\nDone.  Models saved to: {MODEL_DIR}")
