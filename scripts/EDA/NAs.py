import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

PERIOD = "Nocturno"

# ==========================================
# 1. PHYSICAL CONVERSION UTILITIES
# ==========================================

def db_to_pressure(db):
    """Converts Decibels (logarithmic) to Sound Pressure (linear scale)."""
    return 10**(db / 20)

def pressure_to_db(p):
    """Converts Sound Pressure back to Decibels (dB)."""
    # Uses a small floor value to avoid log10(0)
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
    """Inserts rows for missing dates to ensure a continuous daily index."""
    full_range = pd.date_range(start=df['FECHA'].min(), end=df['FECHA'].max(), freq='D')
    df = df.set_index('FECHA')
    
    # Reindexing inserts NaNs where dates were missing in the original file
    df_sync = df.reindex(full_range)
    df_sync.index.name = 'FECHA'
    
    missing_dates = len(full_range) - len(df)
    if missing_dates > 0:
        print(f"Timeline Synchronization: {missing_dates} missing dates inserted as NaNs.")
    
    return df_sync

def apply_integrity_filter(df):
    """Slices the dataframe to start only from the first row with 0 missing values."""
    complete_cases = df.dropna()

    if complete_cases.empty:
        print("Integrity Filter Warning: No rows with complete data found.")
        return pd.DataFrame()

    first_valid_date = complete_cases.index[0]
    print(f"Integrity Filter: Data sequence starts from {first_valid_date.date()}")
    
    return df.loc[first_valid_date:].copy()

def process_and_synchronize_noise_data():
    """Main orchestrator for the data preparation stage."""
    input_file = f'data/processed/LAeq{PERIOD}.csv'
    output_file = f'data/processed/LAeq{PERIOD}Filtrado_Pressure.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    raw_df = load_and_prepare_dataset(input_file)
    synced_df = synchronize_timeline(raw_df)
    filtered_df = apply_integrity_filter(synced_df)
    
    if not filtered_df.empty:
        # Convert to linear pressure scale before saving for the rest of the pipeline
        df_pressure = filtered_df.apply(db_to_pressure)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_pressure.to_csv(output_file, sep=';')
        print(f"Synchronization complete. File saved in Pressure scale.")

# ==========================================
# 3. AUDIT & STATISTICS
# ==========================================

# ==========================================
# 3. AUDIT & STATISTICS
# ==========================================

def missing_values_statistics():
    file_path = f'data/processed/LAeq{PERIOD}.csv'
    if not os.path.exists(file_path): return

    # Cargamos el dataframe filtrado
    df = pd.read_csv(file_path, sep=';', index_col=0)
    total_rows = len(df)

    # Diccionario de Nombres de Estaciones (Completa con los de tu dataset)
    # Esto es lo que dará el "WOW" en la tabla de la tesis
    nombres_estaciones = {
        "1": "Paseo de Recoletos",
        "3": "Plaza del Carmen",
        "4": "Plaza de España",
        "11": "Avda. Ramón y Cajal",
        "24": "Casa de Campo",
        "55": "Urb. Embajada (Barajas)",
        "86": "Tres Olivos"
    }

    # 1. CÁLCULO DE TABLA DE NAs POR ESTACIÓN (Para Tabla 3.1)
    stats_list = []
    for col in df.columns:
        nas_count = df[col].isna().sum()
        pct_nas = (nas_count / total_rows) * 100
        nombre = nombres_estaciones.get(col, "Estación " + col)
        
        stats_list.append({
            "ID": col,
            "Nombre de la Estación": nombre,
            "Cantidad (NAs)": nas_count,
            "Porcentaje (%)": round(pct_nas, 2)
        })

    # Crear DataFrame de estadísticas y ordenar por mayor cantidad de NAs
    df_nas_ranking = pd.DataFrame(stats_list)
    df_nas_ranking = df_nas_ranking.sort_values(by="Cantidad (NAs)", ascending=False)

    print("\n" + "="*85)
    print("TOP ESTACIONES CON MÁS VALORES AUSENTES (PARA TABLA LATEX)")
    print("="*85)
    # Mostramos las top 10 o las que necesites para la tesis
    print(df_nas_ranking.head(10).to_string(index=False))
    print("="*85)

    # 2. CÁLCULO DE DISTRIBUCIÓN DE GAPS (Análisis de Robustez)
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
            "Categoría de Gap": ["Pequeño/Medio (<= 14 días)", "Grande (> 14 días)"],
            "Frecuencia": [gaps_le_14, gaps_gt_14],
            "Porcentaje": [f"{(gaps_le_14/total_gaps)*100:.2f}%", f"{(gaps_gt_14/total_gaps)*100:.2f}%"],
            "Método Asignado": ["Estacional (Contexto Acústico)", "Lineal (Fallback Energético)"]
        }
        
        print("\nRESUMEN METODOLÓGICO DE IMPUTACIÓN")
        print(pd.DataFrame(summary_data).to_string(index=False))
        print(f"\nTotal de eventos de pérdida de datos analizados: {total_gaps}")

# ==========================================
# 4. IMPUTATION PIPELINE
# ==========================================

def run_imputation_pipeline():
    """Executes seasonal and linear imputation in the pressure domain."""
    input_path = f'data/processed/LAeq{PERIOD}Filtrado_Pressure.csv'
    final_output = f'data/processed/LAeq{PERIOD}Final.csv'
    
    if not os.path.exists(input_path): return
    
    df = pd.read_csv(input_path, sep=';', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # STEP 1: Seasonal Imputation (Gap <= 14 days)
    # Using 3-Nearest Neighbors logic based on Day of Week
    print("\n[1/3] Executing Seasonal Imputation (k=3, gap <= 14d)...")
    for col in df.columns:
        is_na = df[col].isna()
        gap_groups = (is_na != is_na.shift()).cumsum()
        
        for _, group_data in df[col][is_na].groupby(gap_groups[is_na]):
            if len(group_data) <= 14:
                for idx in group_data.index:
                    neighbors = []
                    step = 1
                    # Search up to 12 weeks away for neighbors
                    while len(neighbors) < 3 and step < 12:
                        for direction in [-1, 1]:
                            n_idx = idx + pd.Timedelta(days=7 * step * direction)
                            if n_idx in df.index and not np.isnan(df.loc[n_idx, col]):
                                neighbors.append(df.loc[n_idx, col])
                        step += 1
                    
                    if neighbors:
                        df.loc[idx, col] = np.mean(neighbors[:3])

    # STEP 2: Linear Fallback
    # Fills remaining gaps (> 14 days) using linear interpolation in the pressure domain
    print("[2/3] Executing Linear Fallback (Energy-linear interpolation)...")
    df = df.interpolate(method='linear').bfill().ffill()

    # STEP 3: Re-conversion to Decibels
    print("[3/3] Converting results back to Decibels (dB)...")
    df_db = df.apply(pressure_to_db).round(2)
    
    # Save Final Result
    df_db.to_csv(final_output, sep=';')
    print(f"\nPipeline finished successfully.")
    print(f"Final dataset saved to: {final_output}")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Stage 1: Prep and Sync
    # process_and_synchronize_noise_data()
    
    # Stage 2: Audit
    # missing_values_statistics()
    
    # Stage 3: Impute
    run_imputation_pipeline()