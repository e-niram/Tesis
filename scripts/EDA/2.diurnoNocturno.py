import pandas as pd
import os

def load_processed_data(file_path):
    """Carga el archivo procesado asegurando los tipos de datos correctos."""
    df = pd.read_csv(file_path, sep=';')

    df['NMT'] = df['NMT'].astype(int)
    return df

def pivot_noise_metric(df, metric_column):
    """
    Transforma el dataframe: 
    - Filas: FECHA
    - Columnas: NMT (Estaciones)
    - Valores: La métrica seleccionada (Diurno o Nocturno)
    """
    # Pivotamos los datos
    pivoted_df = df.pivot(index='FECHA', columns='NMT', values=metric_column)
    
    # Opcional: Ordenar por fecha para asegurar cronología
    pivoted_df.index = pd.to_datetime(pivoted_df.index)
    pivoted_df = pivoted_df.sort_index()
    
    return pivoted_df

def save_split_file(df, filename):
    """Guarda el dataframe pivotado en la carpeta de procesados."""
    output_dir = "data/processed"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    path = os.path.join(output_dir, filename)
    # Guardamos con index=True porque las fechas están en el índice
    df.to_csv(path, sep=';', index=True)
    print(f"Archivo generado con éxito: {path}")

def generate_split_noise_files():
    """Orquestador para generar los archivos de Diurno y Nocturno."""
    input_file = "data/processed/ruido_processed_full.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: No se encuentra el archivo {input_file}")
        return

    # 1. Cargar
    df = load_processed_data(input_file)
    
    # 2. Procesar Diurno
    df_diurno = pivot_noise_metric(df, 'LAeqDiurno')
    save_split_file(df_diurno, "Diurno.csv")
    
    # 3. Procesar Nocturno
    df_nocturno = pivot_noise_metric(df, 'LAeqNocturno')
    save_split_file(df_nocturno, "Nocturno.csv")


generate_split_noise_files()