"""
04_fetch_api.py — Fetch one day's noise data from the Madrid Open Data API
and append it to data/final/daytime_final.csv & nighttime_final.csv.

Called daily by the GitHub Actions pipeline (after midnight Madrid time).
Downloads the full CSV and reads the last available day by default.

API reference:
  https://datos.madrid.es/api/3/action/datastore_search
  resource_id: 215885-0-contaminacion-ruido

Raw API record example:
  {"NMT": "3", "Año": "2025", "mes": "4", "dia": "11",
   "tipo": "D", "LAeq": "57,4", "L1": "66,6", ...}

Data format differences vs data/final/:
  - Separate Año/mes/dia columns  → merged to FECHA (ISO date)
  - Comma decimal separator       → dot decimal
  - Long format (one row per station-period) → wide (stations as columns)
  - LAeqDiurno = tipo D Laeq directly
  - LAeqNocturno = tipo N Laeq directly
  - Missing stations → seasonal imputation (same logic as 03_handle_missing.py)
"""

import argparse
import sys
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
FINAL_DIR  = REPO_ROOT / "data" / "final"
DAYTIME_F  = FINAL_DIR / "daytime_final.csv"
NIGHTTIME_F = FINAL_DIR / "nighttime_final.csv"

# ── Source constants ───────────────────────────────────────────────────────────
RESOURCE_ID = "215885-0-contaminacion-ruido"
# The CKAN datastore API is frozen (last updated Feb 2026); the CSV is the authoritative source.
CSV_URL = f"https://datos.madrid.es/egob/catalogo/{RESOURCE_ID}.csv"

# Valid station IDs (the 31 stations present in data/final/)
VALID_STATIONS = {
    1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
    24, 25, 26, 27, 28, 29, 30, 31, 47, 48, 50, 54, 55, 86,
}
# Stations explicitly excluded from data/final/ (e.g. Plaza de España — inconsistent coverage)
EXCLUDED_STATIONS = {4}


# ── Physical conversion (mirrors 03_handle_missing.py) ────────────────────────

def db_to_pressure(db: float | np.ndarray) -> float | np.ndarray:
    return 10 ** (db / 20)


def pressure_to_db(p: float | np.ndarray) -> float | np.ndarray:
    return 20 * np.log10(np.where(p > 0, p, 1e-12))


# ── CSV fetch ──────────────────────────────────────────────────────────────────

def _download_csv() -> tuple[pd.DataFrame, str]:
    """
    Download the full CSV and return (DataFrame, year_col_name).
    Raises RuntimeError on network or parse errors.
    """
    try:
        resp = requests.get(CSV_URL, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Could not connect to portal: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"CSV download timed out: {exc}") from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Portal returned HTTP error: {exc}") from exc

    try:
        df = pd.read_csv(StringIO(resp.text), sep=";", dtype=str)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse CSV: {exc}") from exc

    df.columns = [c.strip() for c in df.columns]

    # The year column has been published as both "Año" and "Ano"
    year_col = next(
        (c for c in df.columns if c.strip().lower() in ("año", "ano", "anio", "year")),
        None,
    )
    if year_col is None:
        raise RuntimeError(f"Year column not found in CSV. Columns: {list(df.columns)}")

    return df, year_col


def fetch_last_day(target_date: date | None = None) -> tuple[pd.DataFrame, date]:
    """
    Download the full CSV and return (day_df, resolved_date).

    If *target_date* is given, filter to that date.
    Otherwise, use the last (most recent) date present in the CSV — robust
    against the portal publishing data later than expected.

    Raises RuntimeError when no records are found for the resolved date.
    """
    df, year_col = _download_csv()

    df["_date"] = pd.to_datetime({
        "year":  df[year_col].str.strip(),
        "month": df["mes"].str.strip(),
        "day":   df["dia"].str.strip(),
    }, errors="coerce")

    if target_date is None:
        resolved = df["_date"].max()
        if pd.isna(resolved):
            raise RuntimeError("Could not parse any dates from the CSV.")
        target_date = resolved.date()

    day_df = df[df["_date"].dt.date == target_date].drop(columns=["_date"])

    if day_df.empty:
        raise RuntimeError(
            f"No records found for {target_date} in the published CSV. "
            "The portal may not have published data for this date yet."
        )

    return day_df, target_date


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_raw(raw: pd.DataFrame, target_date: date) -> dict[str, pd.Series]:
    """
    Transform raw API DataFrame → {'daytime': Series, 'nighttime': Series}.

    Each Series is indexed by station_id (int) with float dB values.
    """
    df = raw.copy()

    # Normalise column names (API may return with accents or varying case)
    df.columns = [c.strip() for c in df.columns]

    # Station ID — keep only known valid stations, drop explicit exclusions
    df["NMT"] = pd.to_numeric(df["NMT"], errors="coerce").astype("Int64")
    df = df[df["NMT"].isin(VALID_STATIONS) & ~df["NMT"].isin(EXCLUDED_STATIONS)].copy()

    # Period type — only D (daytime) and N (nighttime); E is not used
    df["tipo"] = df["tipo"].str.strip().str.upper()
    df = df[df["tipo"].isin({"D", "N"})].copy()

    # LAeq: comma decimal → float
    df["LAeq"] = (
        df["LAeq"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["LAeq"] = pd.to_numeric(df["LAeq"], errors="coerce")
    df = df.dropna(subset=["NMT", "LAeq"])

    # --- Daytime: tipo D, Laeq directly -------------------------------------
    day_vals: dict[int, float] = {}
    for nmt, grp in df[df["tipo"] == "D"].groupby("NMT"):
        day_vals[int(nmt)] = float(grp["LAeq"].mean())

    # --- Nighttime: tipo N, Laeq directly -----------------------------------
    night_vals: dict[int, float] = {}
    for nmt, grp in df[df["tipo"] == "N"].groupby("NMT"):
        night_vals[int(nmt)] = float(grp["LAeq"].mean())

    return {
        "daytime":   pd.Series(day_vals,   name=str(target_date)),
        "nighttime": pd.Series(night_vals, name=str(target_date)),
    }


# ── Seasonal imputation (mirrors 03_handle_missing.py) ────────────────────────

def impute_missing_stations(
    new_row: pd.Series,
    historical: pd.DataFrame,
    target_date: date,
    n_neighbors: int = 3,
    max_step: int = 12,
) -> pd.Series:
    """
    For any station missing in *new_row*, look for seasonal neighbors in
    *historical* at offsets of ±7, ±14, ±21, … days and average them.
    Falls back to the column median if no neighbors are found.

    Works in the pressure domain (mirrors 03_handle_missing.py).
    """
    historical_p = historical.apply(db_to_pressure)
    result = new_row.copy()

    for station in historical.columns:
        col_id = str(station)
        if col_id not in result.index or pd.isna(result[col_id]):
            neighbors: list[float] = []
            step = 1
            while len(neighbors) < n_neighbors and step <= max_step:
                for direction in [-1, 1]:
                    neighbor_date = target_date + timedelta(days=7 * step * direction)
                    neighbor_str = str(neighbor_date)
                    if neighbor_str in historical_p.index:
                        val = historical_p.loc[neighbor_str, station]
                        if not np.isnan(val):
                            neighbors.append(val)
                step += 1

            if neighbors:
                result[col_id] = float(
                    pressure_to_db(np.mean(neighbors[:n_neighbors]))
                )
            else:
                # Last resort: column median (in dB)
                result[col_id] = float(historical[station].median())

    return result


# ── Append to final CSVs ───────────────────────────────────────────────────────

def append_to_final(
    period_data: pd.Series,
    csv_path: Path,
    target_date: date,
) -> bool:
    """
    Load the existing final CSV, check whether *target_date* already exists,
    impute any missing stations, and append the new row.

    Returns True if a new row was appended, False if date already present.
    """
    df = pd.read_csv(csv_path, sep=";", index_col=0)
    df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")

    date_str = str(target_date)
    if date_str in df.index:
        print(f"  {csv_path.name}: {date_str} already present — skipping.")
        return False

    # Align new row to existing station columns
    new_row = pd.Series(index=df.columns, dtype=float, name=date_str)
    for station_col in df.columns:
        sid = int(station_col)
        if sid in period_data.index:
            new_row[station_col] = period_data[sid]

    # Impute missing stations using last 60 days as historical context
    recent = df.iloc[-60:]
    new_row = impute_missing_stations(new_row, recent, target_date)

    df = pd.concat([df, new_row.to_frame().T])
    df.to_csv(csv_path, sep=";")
    print(f"  {csv_path.name}: appended {date_str}.")
    return True


# ── Entry point ────────────────────────────────────────────────────────────────

def main(target_date: date | None = None) -> None:
    hint = str(target_date) if target_date else "last available day in CSV"
    print(f"Fetching data for {hint} …")
    try:
        raw, target_date = fetch_last_day(target_date)
    except RuntimeError as exc:
        print(f"WARNING: {exc}")
        print("Skipping append — no data available yet for this date.")
        sys.exit(0)

    print(f"Resolved date: {target_date}")
    cleaned = clean_raw(raw, target_date)

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    append_to_final(cleaned["daytime"],   DAYTIME_F,   target_date)
    append_to_final(cleaned["nighttime"], NIGHTTIME_F, target_date)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch one day from Madrid noise API.")
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="ISO date to fetch (YYYY-MM-DD). Defaults to last available day in the CSV.",
    )
    args = parser.parse_args()
    main(args.date)
