import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import geopandas as gpd

from constants import COORDINATES

def generate_pro_map(csv_path, output_path='data/clustering/madrid_map_nocturno.pdf'):
    results_df = pd.read_csv(csv_path, sep=';')
    df_coords = pd.DataFrame.from_dict(COORDINATES, orient='index', columns=['lat', 'lon']).reset_index()
    df_coords.columns = ['Station_ID', 'lat', 'lon']
    df_map = pd.merge(results_df, df_coords, on='Station_ID')

    geometry = [Point(xy) for xy in zip(df_map.lon, df_map.lat)]
    gdf = gpd.GeoDataFrame(df_map, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(15, 15))
    colors = {0: 'blue', 1: 'orange', 2: 'red', 3: 'purple'}
    
    # Dibujar puntos
    for cluster_id, color in colors.items():
        subset = gdf[gdf['Cluster'] == cluster_id]
        if not subset.empty:
            subset.plot(ax=ax, color=color, label=f'Cluster {cluster_id}', 
                        markersize=150, edgecolor='white', alpha=0.8, zorder=5)

    # 1. AGREGAR NOMBRES (IDs)
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.Station_ID):
        # Desplazamos el texto un poco (offset) para que no tape el punto
        ax.annotate(str(label), xy=(x, y), xytext=(5, 5), 
                    textcoords="offset points", fontsize=10, 
                    fontweight='bold', color='black', zorder=6)

    # Añadir mapa de fondo
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.set_title('Mapa Acústico de Madrid - Análisis por ID', fontsize=18, pad=20)
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
csv_input = 'data/clustering/automatedresults/Results_Nocturno_Original_k3_s51.csv'
generate_pro_map(csv_input)