import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PERIOD='Nocturno'

def db_to_pressure(db):
    """Convierte decibelios a presión acústica (escala lineal)."""
    return 10**(db / 20)

def pressure_to_db(p):
    """Convierte presión acústica de vuelta a decibelios."""
    return 20 * np.log10(np.where(p > 0, p, 1e-12))

def calculate_cluster_means():
    # 1. Configuración de rutas
    mapping_path = f'data/clustering/automatedresults3/Results_{PERIOD}_Original_k3_s49.csv'
    data_path = f'data/processed/LAeq{PERIOD}Final.csv'
    output_csv = f'data/clustering/automatedresults3/Cluster_Means_{PERIOD}.csv'
    
    # Nombres de los dos archivos de salida
    img_individual = f'cluster_{PERIOD}_profiles_individuales.png'
    img_comparativo = f'cluster_{PERIOD}_profiles_superpuesto.png'

    # 2. Carga de datos
    mapping = pd.read_csv(mapping_path, sep=';')
    mapping.columns = ['Station_ID', 'Cluster']
    
    df_data = pd.read_csv(data_path, sep=';')
    df_data['FECHA'] = pd.to_datetime(df_data['FECHA'])
    df_data.set_index('FECHA', inplace=True)

    # 3. Cálculo de medias por cluster
    clusters = sorted(mapping['Cluster'].unique())
    cluster_means_df = pd.DataFrame(index=df_data.index)

    for c in clusters:
        stations_in_cluster = mapping[mapping['Cluster'] == c]['Station_ID'].astype(str).tolist()
        available_cols = [s for s in stations_in_cluster if s in df_data.columns]
        
        if available_cols:
            pressure_data = df_data[available_cols].apply(db_to_pressure)
            mean_pressure = pressure_data.mean(axis=1)
            cluster_means_df[f'Cluster_{c}'] = pressure_to_db(mean_pressure)
            print(f"Cluster {c}: Media calculada con {len(available_cols)} estaciones.")

    # 4. Guardar CSV de resultados
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cluster_means_df.to_csv(output_csv, sep=';')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Azul, Naranja, Verde

    # ==========================================================
    # IMAGEN 1: GRÁFICOS INDIVIDUALES (3 SUBPLOTS)
    # ==========================================================
    fig1, axes = plt.subplots(nrows=len(clusters), ncols=1, figsize=(12, 15), sharex=True)
    
    for i, c in enumerate(clusters):
        col_name = f'Cluster_{c}'
        if col_name in cluster_means_df.columns:
            ax = axes[i]
            ax.plot(cluster_means_df.index, cluster_means_df[col_name], 
                    color=colors[i], lw=1, marker=None, label=f'Media Cluster {c}')
            
            ax.set_ylim(40, 76)
            ax.set_title(f'Perfil Individual: Cluster {c}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Nivel Sonoro (dB)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

    fig1.tight_layout()
    fig1.savefig(img_individual, dpi=300)
    print(f"Imagen 1 guardada: {img_individual}")

    # ==========================================================
    # IMAGEN 2: GRÁFICO SUPERPUESTO (ÚNICA IMAGEN)
    # ==========================================================
    plt.figure(figsize=(12, 7))
    
    for i, c in enumerate(clusters):
        col_name = f'Cluster_{c}'
        if col_name in cluster_means_df.columns:
            plt.plot(cluster_means_df.index, cluster_means_df[col_name], 
                     color=colors[i], lw=1.5, alpha=0.8, label=f'Cluster {c}')
    
    plt.ylim(40, 76)
    plt.title(f'Comparativa de Perfiles Medios por Cluster (Ruido {PERIOD})', fontsize=16, fontweight='bold')
    plt.ylabel('Nivel Sonoro (dB)')
    plt.xlabel('Fecha')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(loc='upper right', ncol=3)
    
    plt.tight_layout()
    plt.savefig(img_comparativo, dpi=300)
    plt.show() # Opcional: mostrar por pantalla
    print(f"Imagen 2 guardada: {img_comparativo}")

if __name__ == "__main__":
    calculate_cluster_means()