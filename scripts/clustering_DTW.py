import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 1. Carga de datos
file_path = 'data/processed/DiurnoImputado.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# Convertimos la columna FECHA a objetos datetime para el eje X
df['FECHA'] = pd.to_datetime(df['FECHA'])
dates = df['FECHA']

# 2. Preparación de datos (solo numéricos)
df_numeric = df.select_dtypes(include=['number'])
X = df_numeric.T.values # Estaciones como filas

# 3. Escalado
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

# 4. Clustering DTW
n_clusters = 2 # Probar solo 3 y 4. Ya con 3 se ven diferencias, con 4 tenemos un cluster con 1 solo individuo
model = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    max_iter=10,
    random_state=42, # Probar con 10 semillas, todos los gráficos NO van en el escrito. Solo se dice que se probaron con 10 semillas y decir que "la agrupación que más se repite es esta, entonces elegimos esta". Es de ver nada más, sobre todo si cambia el número de individuos en cada cluster; si algo se repite, nos quedamos con una de esas semillas
    n_jobs=-1,
    dtw_inertia=True # Calcular este valor 
)
cluster_labels = model.fit_predict(X)

# Lo ideal es mostrar la variabilidad entre clusters (bueno es alto) y variabilidad intra clusters (bueno es bajo)
# Calcularlo para poder comparar, por ejemplo, para decir que 3 es mejor que 4 en una tabla
# Inertia es lo mimso que variabilida

# 5. Visualización con Fechas Reales
fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 15), sharex=True)

for i in range(n_clusters):
    ax = axes[i]
    cluster_series = X[cluster_labels == i]
    
    for series in cluster_series:
        ax.plot(dates, series.ravel(), color='grey', alpha=0.2)
    
    # Dibujar el centroide
    ax.plot(dates, model.cluster_centers_[i].ravel(), color='red', linewidth=2)
    
    ax.set_title(f'Cluster {i} - Estaciones: {len(cluster_series)}')
    ax.set_ylabel('Decibeles')
    
    # Formatear el eje X para que muestre fechas
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.grid(True, alpha=0.2)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Guardar resultados
station_ids = df_numeric.columns
results = pd.DataFrame({'Station_ID': station_ids, 'Cluster': cluster_labels})
print(results)