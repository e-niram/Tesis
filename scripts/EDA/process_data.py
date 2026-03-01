import os
import pandas as pd

def save_to_csv(df, filename="processed_data.csv"):
    """Saves the dataframe as a CSV file in the data/processed directory."""
    output_dir = "data/processed"
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False, sep=';')
    print(f"File successfully saved to: {path}")

def select_final_columns(df):
    """Keeps only the specific columns requested."""
    target_cols = ['FECHA', 'NMT', 'LAeqDiurno', 'LAeqNocturno']
    return df[target_cols]

def clean_data(df):
    """
    Orchestrates the data cleaning process and saves the output.
    """
    df = create_date_column(df)
    df = pivot_and_rename_types(df)
    df = flatten_columns(df)
    df = convert_numeric_values(df)
    df = select_final_columns(df)
    
    # Final sort
    df = df.sort_values(["FECHA"]).reset_index(drop=True)
    
    # Save the result
    # save_to_csv(df, f"ruido_processed.csv")
    
    return df

def filter_by_station(df, station_code):
    """Filters the dataframe by a specific station code."""
    return df[df['NMT'].isin([station_code])]

def create_date_column(df):
    """Combines Year, Month, and Day columns into a single datetime column."""
    df["FECHA"] = pd.to_datetime(
        {
            "year": df["Año"],
            "month": df["mes"],
            "day": df["dia"]
        },
        errors="coerce"
    )
    # Move FECHA to the first position
    cols = ['FECHA'] + [col for col in df.columns if col != 'FECHA']
    df = df.reindex(columns=cols)
    # Drop original date components
    df.drop(columns=['Año', 'mes', 'dia'], inplace=True)
    return df

def pivot_and_rename_types(df):
    """Pivots the 'tipo' column and renames the resulting time periods."""
    df = (
        df
        .pivot(index=["FECHA", "NMT"], columns="tipo", 
               values=["LAeq", "L1", "L10", "L50", "L90", "L99"])
        .reset_index()
    )
    
    # Rename the sub-columns from the 'tipo' pivot
    df = df.rename(columns={
        "D": "Diurno",
        "E": "Vespertino",
        "N": "Nocturno",
        "T": "Total"
    })
    return df

def flatten_columns(df):
    """Converts multi-index columns into a single string (e.g., L90Diurno)."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            main, sub = col
            new_cols.append(f"{main}{sub}") # e.g., "L90" + "Diurno" → "L90Diurno"
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

def convert_numeric_values(df):
    """Replaces commas with dots and converts object columns to floats."""
    # We skip 'FECHA' and 'NMT' to avoid errors during conversion
    for col in df.columns:
        if df[col].dtype == object and col not in ["FECHA", "NMT"]:
            df[col] = df[col].str.replace(",", ".").astype(float)
    return df

def save_to_csv(df, filename="processed_data.csv"):
    """Saves the dataframe as a CSV file in the data/processed directory."""
    output_dir = "data/processed"
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False, sep=';')
    print(f"File successfully saved to: {path}")

# Usage example:
# df_cleaned = clean_data(original_df, "14")


import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("data/raw/ruido.csv", delimiter=";", on_bad_lines='warn')
clean_data(df)