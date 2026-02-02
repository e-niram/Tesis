import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 1. Load the dataset from Excel
file_path = 'data/processed/DiurnoImputado.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 2. Data Preparation
# We must ensure ONLY numeric columns are transposed. 
# We filter by numeric types to safely exclude 'FECHA' and any other non-numeric info.
df_numeric = df.select_dtypes(include=['number'])

# Transpose: rows become stations (Time Series)
X = df_numeric.T.values

# 3. Scaling (Mean-Variance)
# Now X contains only floats, so the error will be resolved.
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

# 4. DTW Clustering
n_clusters = 4
model = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    max_iter=10,
    random_state=42,
    n_jobs=-1 # Use all CPU cores to speed up DTW calculation
)

cluster_labels = model.fit_predict(X_scaled)

# 5. Mapping Results
# We use the column names from df_numeric to identify the stations
station_ids = df_numeric.columns
results = pd.DataFrame({
    'Station_ID': station_ids,
    'Cluster': cluster_labels
})

# 6. Visualization
plt.figure(figsize=(12, 10))
for i in range(n_clusters):
    plt.subplot(n_clusters, 1, i + 1)
    cluster_series = X_scaled[cluster_labels == i]
    for series in cluster_series:
        plt.plot(series.ravel(), color='grey', alpha=0.3)
    
    # Plot the centroid (the average shape of the cluster)
    plt.plot(model.cluster_centers_[i].ravel(), color='red', linewidth=2)
    plt.title(f'Cluster {i} - Member Stations: {len(cluster_series)}')
    plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print(results)