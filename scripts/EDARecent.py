import matplotlib.pyplot as plt
import pandas as pd
import os

# Definition of STATIONS (Ensure this matches your needs)
STATIONS = {
    1: "Paseo de Recoletos", 2: "Carlos V", 3: "Plaza del Carmen", 4: "Plaza de España",
    5: "Barrio del Pilar", 6: "Pascual Rodríguez", 8: "Campo de las Naciones",
    10: "Castellana", 11: "Méndez Álvaro", 12: "Villaverde", 13: "Vallecas",
    14: "Moratalaz", 16: "Arturo Soria", 17: "Villaverde Alto", 18: "Farolillo",
    19: "Huerta Castañeda", 20: "Moratalaz Alto", 24: "Casa de Campo",
    25: "Barajas", 26: "Ventas", 27: "Plaza de Castilla", 28: "Cuatro Vientos",
    29: "El Pardo", 30: "Sanchinarro", 31: "Ensanche de Vallecas",
    47: "Mendez Alvaro II", 48: "Castellana II", 50: "Plaza de Castilla II",
    54: "Ensanche de Vallecas II", 55: "Urb. Embajada", 86: "Tres Olivos"
}

def plot_excel_time_series():
    file_path = "data/processed/DiurnoReciente2.xlsx"
    output_dir = "plots/EDA"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load Excel
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Standardize Column Names: Convert all column names to strings to avoid type mismatch
    df.columns = [str(col) for col in df.columns]
    
    # Ensure FECHA is datetime
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"])
    else:
        print("Error: Column 'FECHA' not found.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for station_id, name in STATIONS.items():
        # Search for the column as a string since we converted df.columns to strings
        col_name = str(station_id)
        
        if col_name in df.columns:
            print(f"Generating plot for Station: {station_id} - {name}")
            
            # Extract data and remove NaNs for this specific station
            df_station = df[["FECHA", col_name]].dropna()
            
            if not df_station.empty:
                save_station_plot(df_station, station_id, name, col_name, output_dir)
            else:
                print(f"   Warning: Station {station_id} has no valid data points.")
        else:
            print(f"   Skip: Column '{col_name}' not found in Excel file.")

def save_station_plot(df, station_id, station_name, col_name, output_dir):
    plt.figure(figsize=(12, 6))
    
    plt.plot(df["FECHA"], df[col_name], label="LAeq Diurno", color='tab:orange', linewidth=1.5, alpha=0.9)
    
    plt.ylim(30, 105) 
    plt.xlabel("Fecha")
    plt.ylabel("Decibeles (dB)")
    plt.title(f"Evolución Diaria Nivel de Ruido Diurno: {station_name}")
    
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filename = f"station_{station_id}_diurno_recent_november.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_excel_time_series()