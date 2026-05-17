"""
Rolling naive forecast (seasonal naive, period=7).

For each window, the forecast for step h (h=1..lead) is the value
observed at position (h-1) % 7 steps back from the end of the last
full week in the training data — i.e. the same weekday from the
prior week. Mirrors the expanding-window logic in predictions.sas
(train_obs=3500, lead=14).

Output: sas/results/forecast_metrics_{period}_{cluster_id}_naive.csv
        Columns match the SAS ESM/ARIMA output so compare_models.py
        can ingest them without changes.
"""
import math
import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np

TRAIN_OBS = 3500
LEAD      = 14
DATE_CAP  = '2025-12-31'

CLUSTER_MEANS = {
    'daytime':   'results/clustering/final/metrics/Cluster_Means_daytime.csv',
    'nighttime': 'results/clustering/final/metrics/Cluster_Means_nighttime.csv',
}

OUT_DIR = Path('sas/results')


def sas_date(d: pd.Timestamp) -> str:
    return d.strftime('%d%b%Y').upper()


def rolling_naive(series: pd.Series, train_obs: int = TRAIN_OBS,
                  lead: int = LEAD):
    """Yield per-window metrics dicts over an expanding window."""
    n = len(series)
    current_end = train_obs
    window_num  = 1

    while current_end + lead <= n:
        # Forecast: repeat the last `lead` days of the prior week twice.
        # For step h (1-based), naive = series[current_end - 7 + (h-1) % 7]
        forecasts = []
        actuals   = []

        for h in range(1, lead + 1):
            lag_idx    = current_end - 7 + (h - 1) % 7   # 0-based
            forecast   = series.iloc[lag_idx]
            actual     = series.iloc[current_end + h - 1]
            forecasts.append(forecast)
            actuals.append(actual)

        errors     = [a - f for a, f in zip(actuals, forecasts)]
        abs_errors = [abs(e) for e in errors]
        sq_errors  = [e ** 2 for e in errors]
        pct_errors = [abs(e / a) * 100 for e, a in zip(errors, actuals) if a != 0]

        mae  = np.mean(abs_errors)
        rmse = math.sqrt(np.mean(sq_errors))
        mape = np.mean(pct_errors) if pct_errors else float('nan')

        dates = series.index[current_end: current_end + lead]

        yield {
            'window_num':      window_num,
            'train_end_obs':   current_end,
            'forecast_start':  sas_date(dates[0]),
            'forecast_end':    sas_date(dates[-1]),
            'n':               lead,
            'MAE':             round(mae,  4),
            'RMSE':            round(rmse, 4),
            'MAPE':            round(mape, 4),
        }

        current_end += lead
        window_num  += 1


def run(period: str, data_path: str):
    df = pd.read_csv(data_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df.set_index('FECHA', inplace=True)
    df = df[df.index <= DATE_CAP]

    clusters = [c for c in df.columns if c.startswith('Cluster_')]

    for col in clusters:
        cluster_id = col.lower()   # e.g. cluster_0
        series     = df[col].dropna()

        rows = list(rolling_naive(series))

        out_path = OUT_DIR / f'forecast_metrics_{period}_{cluster_id}_naive.csv'
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f'[{period}] {cluster_id}: {len(rows)} windows → {out_path}')


if __name__ == '__main__':
    for period, path in CLUSTER_MEANS.items():
        run(period, path)
