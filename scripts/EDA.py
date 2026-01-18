import matplotlib.pyplot as plt
import pandas as pd
import os
from constants import STATIONS

def plot_processed_data():
    """Iterates through all stations, loads their data, and generates plots."""
    file_path = "data/processed/ruido_processed.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    df_full = pd.read_csv(file_path, sep=';')
    df_full["FECHA"] = pd.to_datetime(df_full["FECHA"])

    for station_id, name in STATIONS.items():
        print(f"Processing Station: {station_id} - {name}")
        
        # Filter data for this specific station
        df_station = df_full[df_full['NMT'] == station_id]
        
        if not df_station.empty:
            # 1. Plot full historical data
            plot_station(df_station, station_id, name)
            
            # 2. Plot only from 2022 onwards
            df_2022 = df_station[df_station["FECHA"] >= "2022-01-01"]
            if not df_2022.empty:
                plot_station(df_2022, station_id, name, suffix="2022")
        else:
            print(f"   No data found for station {station_id}")

def plot_station(df, station_id, station_name, suffix=""):
    """Generates and saves a time series plot for a specific station."""
    plt.figure(figsize=(12, 6))
    
    plt.plot(df["FECHA"], df["LAeqDiurno"], label="LAeq Diurno", color='tab:blue', alpha=0.8)
    plt.plot(df["FECHA"], df["LAeqNocturno"], label="LAeq Nocturno", color='tab:orange', alpha=0.8)
    
    title_extra = " (Desde 2022)" if suffix == "2022" else ""
    plt.xlabel("Fecha")
    plt.ylabel("Decibeles (dB)")
    plt.title(f"Evolución Nivel de Ruido: {station_name}{title_extra}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Ensure the directory exists
    output_dir = "plots/EDA"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save with custom suffix if provided
    filename = f"station_{station_id}_{suffix}.png" if suffix else f"station_{station_id}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

plot_processed_data()