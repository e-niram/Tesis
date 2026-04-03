import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.constants import STATIONS

# ==========================================
# 1. PHYSICAL CONVERSION UTILITIES
# ==========================================

def db_to_pressure(db):
    """Converts Decibels (logarithmic) to Sound Pressure (linear scale)."""
    return 10**(db / 20)

def pressure_to_db(p):
    """Converts Sound Pressure back to Decibels (dB)."""
    return 20 * np.log10(np.where(p > 0, p, 1e-12))

# ==========================================
# 2. DATA PREPARATION & SYNCHRONIZATION
# ==========================================

def load_and_prepare_dataset(file_path):
    """Loads the CSV and ensures correct DateTime formatting."""
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    return df

def synchronize_timeline(df):
    """Inserts rows for missing dates to ensure a continuous daily index, excluding 2026."""
    # 1. Define the initial range
    start_date = df['FECHA'].min()
    end_date = df['FECHA'].max()
    
    # 2. Force the end_date to not exceed the last day of 2025
    limit_date = pd.Timestamp('2025-12-31')
    if end_date > limit_date:
        end_date = limit_date
        print(f"Timeline Clip: Data restricted to end at {limit_date.date()}")

    full_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 3. Filter the original dataframe to remove any 2026 rows before reindexing
    df = df[df['FECHA'] <= limit_date]
    df = df.set_index('FECHA')
    
    # 4. Reindex to the cleaned range
    df_sync = df.reindex(full_range)
    df_sync.index.name = 'FECHA'
    
    missing_dates = len(full_range) - len(df)
    if missing_dates > 0:
        print(f"Timeline Synchronization: {missing_dates} missing dates inserted.")
    
    return df_sync

def apply_integrity_filter(df):
    """Slices the dataframe to start only from the first row with 0 missing values."""
    complete_cases = df.dropna()
    if complete_cases.empty:
        return pd.DataFrame()

    first_valid_date = complete_cases.index[0]
    return df.loc[first_valid_date:].copy()

def process_and_synchronize_noise_data(period):
    """Main orchestrator for the data preparation stage."""
    input_file = f'data/processed/{period}.csv'
    output_file = f'data/processed/{period}_pressure.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    raw_df = load_and_prepare_dataset(input_file)
    synced_df = synchronize_timeline(raw_df)
    filtered_df = apply_integrity_filter(synced_df)
    
    if not filtered_df.empty:
        df_pressure = filtered_df.apply(db_to_pressure)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_pressure.to_csv(output_file, sep=';')
        print(f"[{period}] Synchronization complete. Saved in Pressure scale.")

# ==========================================
# 3. AUDIT & STATISTICS
# ==========================================

def missing_values_statistics(period):
    file_path = f'data/processed/{period}.csv'
    if not os.path.exists(file_path): return

    df = pd.read_csv(file_path, sep=';', index_col=0)
    total_rows = len(df)

    stats_list = []
    for col in df.columns:
        nas_count = df[col].isna().sum()
        pct_nas = (nas_count / total_rows) * 100
        # Use imported STATIONS constant
        nombre = STATIONS.get(col, f"Estación {col}")
        
        stats_list.append({
            "ID": col,
            "Nombre de la Estación": nombre,
            "Cantidad (NAs)": nas_count,
            "Porcentaje (%)": round(pct_nas, 2)
        })

    df_nas_ranking = pd.DataFrame(stats_list).sort_values(by="Cantidad (NAs)", ascending=False)

    print(f"\n{'='*30} {period.upper()} STATS {'='*30}")
    print(df_nas_ranking.head(10).to_string(index=False))

    all_gap_lengths = []
    for col in df.columns:
        is_na = df[col].isna()
        gap_groups = (is_na != is_na.shift()).cumsum()
        gaps = is_na[is_na].groupby(gap_groups[is_na]).sum().tolist()
        all_gap_lengths.extend(gaps)
    
    if all_gap_lengths:
        total_gaps = len(all_gap_lengths)
        gaps_le_14 = len([g for g in all_gap_lengths if g <= 14])
        gaps_gt_14 = len([g for g in all_gap_lengths if g > 14])
        
        summary_data = {
            "Categoría": ["<= 14 días", "> 14 días"],
            "Frecuencia": [gaps_le_14, gaps_gt_14],
            "Método": ["Estacional (k=3)", "Lineal (Pressure)"]
        }
        print(pd.DataFrame(summary_data).to_string(index=False))

# ==========================================
# 4. IMPUTATION PIPELINE
# ==========================================

def run_imputation_pipeline(period):
    """Executes seasonal and linear imputation in the pressure domain."""
    input_path = f'data/processed/{period}_pressure.csv'
    final_output = f'data/final/{period}_final.csv'
    
    if not os.path.exists(input_path): return
    
    df = pd.read_csv(input_path, sep=';', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    print(f"\n--- Processing Imputation for {period} ---")
    
    # STEP 1: Seasonal Imputation (Gap <= 14 days)
    for col in df.columns:
        is_na = df[col].isna()
        gap_groups = (is_na != is_na.shift()).cumsum()
        
        for _, group_data in df[col][is_na].groupby(gap_groups[is_na]):
            if len(group_data) <= 14:
                for idx in group_data.index:
                    neighbors = []
                    step = 1
                    while len(neighbors) < 3 and step < 12:
                        for direction in [-1, 1]:
                            n_idx = idx + pd.Timedelta(days=7 * step * direction)
                            if n_idx in df.index and not np.isnan(df.loc[n_idx, col]):
                                neighbors.append(df.loc[n_idx, col])
                        step += 1
                    if neighbors:
                        df.loc[idx, col] = np.mean(neighbors[:3])

    # STEP 2: Linear Fallback
    df = df.interpolate(method='linear').bfill().ffill()

    # STEP 3: Re-conversion to Decibels
    df_db = df.apply(pressure_to_db).round(2)
    df_db.to_csv(final_output, sep=';')
    print(f"Pipeline finished for {period}. Result saved.")

# ==========================================
# MAIN EXECUTION (AUTOMATED)
# ==========================================

if __name__ == "__main__":
    PERIODS = ["daytime", "nighttime"]
    
    for p in PERIODS:
        print(f"\n{'#'*50}\n# STARTING PROCESS FOR: {p.upper()}\n{'#'*50}")
        
        # Step 1: Prep
        process_and_synchronize_noise_data(p)
        
        # Step 2: Statistics
        missing_values_statistics(p)
        
        # Step 3: Imputation
        run_imputation_pipeline(p)