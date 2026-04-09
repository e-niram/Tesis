import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import geopandas as gpd

from src.constants import COORDINATES

def _db_to_pressure(db):
    return 10 ** (db / 20)

def _pressure_to_db(p):
    return 20 * np.log10(np.where(p > 0, p, 1e-12))

def generate_pro_map(csv_path, data_path, output_path='results/clustering/madrid_map_nighttime.pdf'):
    results_df = pd.read_csv(csv_path, sep=';')
    df_coords = pd.DataFrame.from_dict(COORDINATES, orient='index', columns=['lat', 'lon']).reset_index()
    df_coords.columns = ['Station_ID', 'lat', 'lon']
    df_map = pd.merge(results_df, df_coords, on='Station_ID')

    # Rank clusters by mean noise level: lowest→green, middle→yellow, highest→red
    df_data = pd.read_csv(data_path, sep=';', index_col=0)
    clusters = sorted(results_df['Cluster'].unique())
    overall_means = {}
    for c in clusters:
        stations = results_df[results_df['Cluster'] == c]['Station_ID'].astype(str).tolist()
        available = [s for s in stations if s in df_data.columns]
        if available:
            mean_pressure = df_data[available].apply(_db_to_pressure).mean(axis=1)
            overall_means[c] = _pressure_to_db(mean_pressure).mean()
    rank = sorted(overall_means, key=overall_means.get)  # ascending
    color_map = {rank[0]: 'green', rank[1]: '#CC8400', rank[2]: 'red'}
    label_map = {rank[0]: 'Ruido bajo', rank[1]: 'Ruido medio', rank[2]: 'Ruido alto'}
    display_order = [rank[2], rank[1], rank[0]]  # Red → Yellow → Green

    geometry = [Point(xy) for xy in zip(df_map.lon, df_map.lat)]
    gdf = gpd.GeoDataFrame(df_map, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(15, 15))

    # Dibujar puntos
    for cluster_id in display_order:
        subset = gdf[gdf['Cluster'] == cluster_id]
        if not subset.empty:
            subset.plot(ax=ax, color=color_map[cluster_id], label=label_map[cluster_id],
                        markersize=150, edgecolor='white', alpha=0.8, zorder=5)

    # 1. AGREGAR NOMBRES (IDs)
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.Station_ID):
        # Desplazamos el texto un poco (offset) para que no tape el punto
        ax.annotate(str(label), xy=(x, y), xytext=(5, 5), 
                    textcoords="offset points", fontsize=10, 
                    fontweight='bold', color='black', zorder=6)

    # Añadir mapa de fondo
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    period_label = 'Nocturno' if 'nighttime' in output_path else 'Diurno'
    ax.set_title(f'Mapa Acústico de Madrid - Ruido {period_label}', fontsize=18, pad=20)
    ax.axis('off')
    ax.legend(title="Grupos Acústicos", loc='lower right', frameon=True, fontsize=12)

    # 2. GUARDAR EN FORMATO VECTORIAL (PDF o SVG)
    plt.tight_layout()
    # Guardamos en PDF para resolución infinita en la tesis
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    # También guardamos un PNG de muy alta resolución (400 DPI) por si acaso
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=400, bbox_inches='tight')
    
    plt.close()
    print(f"Archivos guardados: {output_path} (Vectorial) y versión PNG.")

# Ejecución
MAPS = {
    'nighttime': 'results/clustering/metrics/Results_nighttime_k3_s47.csv',
    'daytime':   'results/clustering/metrics/Results_daytime_k3_s48.csv',
}

for period, csv_input in MAPS.items():
    generate_pro_map(
        csv_path=csv_input,
        data_path=f'data/final/{period}_final.csv',
        output_path=f'results/clustering/madrid_map_{period}.pdf',
    )