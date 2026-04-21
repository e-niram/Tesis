"""
06_train_and_save_models.py — Validate Holt-Winters Additive on all stations.

With a fixed model structure (Holt-Winters Additive, seasonal_periods=7) there
is nothing to pre-train: ES parameters are fitted per station at inference time
in 05_update_predictions.py.

This script runs a dry-fit on every station series and prints a summary of AIC
values and any stations that fall back to level-only smoothing due to sparse
data.  Called monthly by GitHub Actions as a quality check.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

REPO_ROOT  = Path(__file__).resolve().parents[2]
FINAL_DIR  = REPO_ROOT / "data" / "final"

FINAL_CSVS = {
    "daytime":   FINAL_DIR / "daytime_final.csv",
    "nighttime": FINAL_DIR / "nighttime_final.csv",
}

ES_KWARGS = dict(trend="add", damped_trend=False, seasonal="add", seasonal_periods=7)
MIN_OBS   = 14


def validate_period(period: str) -> None:
    df = pd.read_csv(FINAL_CSVS[period], sep=";", index_col=0)
    df.columns = df.columns.astype(str)

    print(f"\n── {period} ({len(df)} observations) ──")
    print(f"  {'Station':<8} {'n_obs':>6} {'model':<12} {'AIC':>10}")
    print("  " + "-" * 40)

    for col in df.columns:
        series = df[col].dropna().to_numpy(dtype=float)
        n = len(series)
        if n < MIN_OBS:
            print(f"  {col:<8} {n:>6} {'level-only':<12} {'—':>10}  ⚠ sparse")
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(series, **ES_KWARGS).fit(optimized=True, disp=False)
            print(f"  {col:<8} {n:>6} {'HW-additive':<12} {fit.aic:>10.1f}")
        except Exception as exc:
            print(f"  {col:<8} {n:>6} {'FAILED':<12} {'—':>10}  ✗ {exc}")


if __name__ == "__main__":
    for period in ("daytime", "nighttime"):
        validate_period(period)
    print("\nValidation complete.")
