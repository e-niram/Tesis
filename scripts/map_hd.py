import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import geopandas as gpd

from constants import COORDINATES

def generate_uhd_map(csv_path, output_path='data/clustering/madrid_map_UHD.png'):
    # 1. Preparación de datos (Igual que antes)
    results_df = pd.read_csv(csv_path, sep=';')
    df_coords = pd.DataFrame.from_dict(COORDINATES, orient='index', columns=['lat', 'lon']).reset_index()
    df_coords.columns = ['Station_ID', 'lat', 'lon']
    df_map = pd.merge(results_df, df_coords, on='Station_ID')

    geometry = [Point(xy) for xy in zip(df_map.lon, df_map.lat)]
    gdf = gpd.GeoDataFrame(df_map, geometry=geometry, crs="EPSG:4326").to_crs(epsg=3857)

    # 2. Configuración del lienzo para Alta Resolución
    # Reducimos el tamaño físico (figsize) pero aumentamos los DPI al guardar
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = {0: 'blue', 1: 'orange', 2: 'red', 3: 'purple'}
    
    # Dibujar puntos
    for cluster_id, color in colors.items():
        subset = gdf[gdf['Cluster'] == cluster_id]
        if not subset.empty:
            subset.plot(ax=ax, color=color, label=f'Cluster {cluster_id}', 
                        markersize=100, edgecolor='white', linewidth=1.5, alpha=0.9, zorder=5)

    # Añadir etiquetas de ID
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.Station_ID):
        ax.annotate(str(label), xy=(x, y), xytext=(4, 4), 
                    textcoords="offset points", fontsize=9, 
                    fontweight='bold', zorder=6)

    # 3. EL TRUCO PARA LA RESOLUCIÓN: Parámetro 'zoom'
    # 'zoom' fuerza a descargar una capa con más detalle. 
    # Para Madrid, un valor entre 13 y 15 suele ser perfecto.
    try:
        ctx.add_basemap(ax, 
                        source=ctx.providers.CartoDB.Positron, 
                        zoom=14,  # <-- Aumenta este número para más detalle (cuidado: tarda más en descargar)
                        interpolation='sinc') # Mejora la calidad del re-escalado
    except Exception as e:
        print(f"Error con el zoom seleccionado: {e}. Intentando modo automático...")
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    ax.axis('off')
    ax.legend(title="Grupos Acústicos", loc='lower right', frameon=True)

    # 4. GUARDADO DE ALTA DENSIDAD
    plt.tight_layout()
    # Guardamos a 600 DPI. Esto generará una imagen muy pesada pero nítida al hacer zoom.
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"Mapa de alta resolución guardado en: {output_path}")

# Ejecución
csv_input = 'data/clustering/automatedresults/Results_Diurno_Original_k2_s44.csv'
generate_uhd_map(csv_input)