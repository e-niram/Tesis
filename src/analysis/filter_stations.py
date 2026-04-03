import pandas as pd
import os

def remove_station(station_id, input_path='data/processed/LAeqDiurnoFiltrado_Pressure.csv', output_path=None):
    """
    Removes a specific station from the dataset to prevent using synthetic data.
    """
    # Verify file existence
    if not os.path.exists(input_path):
        print(f"Error: El archivo {input_path} no existe.")
        return

    # 1. Load the dataset
    # We maintain the FECHA index to preserve time series structure
    df = pd.read_csv(input_path, sep=';', index_col=0)
    
    # Ensure ID is handled as a string
    col_name = str(station_id)

    # 2. Check existence and drop the column
    if col_name in df.columns:
        df_cleaned = df.drop(columns=[col_name])
        print(f"La estación {col_name} ha sido eliminada exitosamente.")
        
        # 3. Save changes
        save_path = output_path if output_path else input_path
        df_cleaned.to_csv(save_path, sep=';')
        print(f"Dataset actualizado guardado en: {save_path}")
        print(f"Dimensiones finales del dataset: {df_cleaned.shape}")
    else:
        print(f"Aviso: La estación {col_name} no se encontró en el archivo.")

# ==========================================
# EXECUTION (Example usage)
# ==========================================
if __name__ == "__main__":
    # Path to the processed data
    input_file = 'data/processed/LAeqNocturnoFinal.csv'
    
    # Example: Removing 'Plaza de España' (ID 4) as discussed in your audit
    remove_station('4', input_path=input_file)