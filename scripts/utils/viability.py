import pandas as pd

def scan_window_viability(file_path):
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    stations = [col for col in df.columns if col != 'FECHA']
    
    # Umbrales definidos metodológicamente
    threshold_cubic = 5
    threshold_seasonal = 14
    
    years_to_test = [2020, 2021, 2022, 2023, 2024, 2025]
    
    # Encabezado de la tabla (ajustado para incluir IDs)
    header = f"{'Año Inicio':<10} | {'G1 (Cúb.)':<10} | {'G2 (Stac.)':<10} | {'G3 (Desc.)':<10} | {'Total Aptas':<12} | {'IDs Descartados (G3)'}"
    print(header)
    print("-" * 100)

    for year in years_to_test:
        df_year = df[df['FECHA'].dt.year >= year].copy()
        
        if df_year.empty:
            continue
            
        g1_count = 0 
        g2_count = 0 
        g3_ids = []  # Lista para almacenar los IDs del Grupo 3
        
        for col in stations:
            is_na = df_year[col].isna()
            
            if not is_na.any():
                g1_count += 1
                continue
                
            gap_groups = (is_na != is_na.shift()).cumsum()
            max_gap = is_na[is_na].groupby(gap_groups[is_na]).sum().max()
            
            if max_gap <= threshold_cubic:
                g1_count += 1
            elif max_gap <= threshold_seasonal:
                g2_count += 1
            else:
                g3_ids.append(col)
                
        g3_count = len(g3_ids)
        aptas = g1_count + g2_count
        # Convertimos la lista de IDs a string separado por comas
        ids_str = ", ".join(g3_ids) if g3_ids else "Ninguna"
        
        print(f"{year:<10} | {g1_count:<10} | {g2_count:<10} | {g3_count:<10} | {aptas:<12} | {ids_str}")

# Ejecución
scan_window_viability('data/processed/LAeqNocturnoFiltrado.csv')


Menos de 2 semanas --> Cúbica
Más de 2 semanas --> Promediar periodo anterior y posterior --> Poner esto como una innovación en el escrito, porque no está en SAS y es muy razonable cuando hay un comportamiendo periodico que se quiere reproducir
Más del 20% de los datos ausentes --> Descartar

Si no puedo detectar la interpolación (e.g. estacoin 24), entonces dejamos la cúbica

Tomar las 2 semanas anteriores, las 2 semanas posteriores, y:

1. Rellenear con los mismos datos
2. Media por día (e.g. media de los lunes) y ponerlo en los lunes que me faltan

Ej: me faltan 2 semanas
2 semanas anteriores y 1 posterior para la primera semana que falta
1 semana anterior y 2 posteriores para la segunda que falta

Si me faltan 2 meses, hago lo mismo pero tomando meses anteriores y meses posteriores
E.g. me falta marzo 2024, entonces promedio marzo 2023 y 2025

Conviene no repetir el mismo valor en varios sitios, porque imputar por la media disminuye la variabilidad de la muestra

Si faltan demasiados datos, eliminar la serie y justificar por qué (e.g. obras)
Ej: imputar el 20% de los datos es demasiado
Menos del 20% sí se puede imputar