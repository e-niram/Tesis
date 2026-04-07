import pandas as pd


HIGH_NA_THRESHOLD = 0.20


def filter_stations_by_na_threshold(df: pd.DataFrame, threshold: float = HIGH_NA_THRESHOLD) -> pd.DataFrame:
    """
    Drops columns whose proportion of NaN values exceeds the given threshold.

    Should be applied to the raw (pre-imputation) pressure DataFrame so that
    the threshold reflects real missing data, not imputed values.

    Returns the filtered DataFrame and prints which stations were removed.
    """
    na_ratios = df.isna().mean()
    to_drop = na_ratios[na_ratios > threshold].index.tolist()

    if to_drop:
        print(f"Stations removed (>{threshold*100:.0f}% NAs): {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        print(f"No stations exceeded the {threshold*100:.0f}% NA threshold.")

    return df
