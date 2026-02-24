import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 1. Setup paths and directories
base_dir = 'data/clustering'
plots_dir = os.path.join(base_dir, 'automatedplots')
results_dir = os.path.join(base_dir, 'automatedresults')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 2. Load and Prepare Data
file_path = 'data/processed/NocturnoImputado.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
df['FECHA'] = pd.to_datetime(df['FECHA'])
dates = df['FECHA']

df_numeric = df.select_dtypes(include=['number'])
station_ids = df_numeric.columns
X_orig = df_numeric.T.values
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_orig)

# List for capturing inertia results
inertia_records = []

# 3. Execution Loop
# Settings for 2x3x10 iterations
# data_modes = {'Scaled': X_scaled, 'Original': X_orig}
data_modes = {'Original': X_orig}
# n_clusters_list = [2, 3, 4]
n_clusters_list = [4]
seeds = range(42, 52) # 10 different seeds

for mode_name, data in data_modes.items():
    print(f"Processing mode: {mode_name}...")
    
    for n_clusters in n_clusters_list:
        for seed in seeds:
            run_id = f"Nocturno_{mode_name}_k{n_clusters}_s{seed}"
            
            # Initialize and fit model
            model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric="dtw", # Probar con distancia euclidea
                max_iter=10,
                random_state=seed,
                n_jobs=-1,
                dtw_inertia=True
            )
            
            cluster_labels = model.fit_predict(data)
            current_inertia = model.inertia_
            
            # A. Save Inertia to list
            inertia_records.append({
                'Mode': mode_name,
                'K': n_clusters,
                'Seed': seed,
                'Inertia': current_inertia
            })
            
            # B. Save Station Results (CSV is better for data integrity)
            results = pd.DataFrame({'Station_ID': station_ids, 'Cluster': cluster_labels})
            results.to_csv(f"{results_dir}/Results_{run_id}.csv", index=False, sep=';')
            
            # C. Generate and Save Plot
            fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 4*n_clusters), sharex=True)
            if n_clusters == 1: axes = [axes] # Handle single plot case if k=1 was used
            
            for i in range(n_clusters):
                ax = axes[i]
                cluster_series = data[cluster_labels == i]
                
                for series in cluster_series:
                    ax.plot(dates, series.ravel(), color='grey', alpha=0.2)
                
                ax.plot(dates, model.cluster_centers_[i].ravel(), color='red', linewidth=2)
                ax.set_title(f'Clúster {i} - Cantidad de Estaciones: {len(cluster_series)}')
                ax.set_ylabel('Desviaciones con respecto a la media' if mode_name == 'Scaled' else 'Decibeles (dB)')

                # Apply fixed scale only to Original (dB) data
                if mode_name == 'Original':
                    ax.set_ylim([30, 105])
                else:
                    # For Scaled data, a range of -4 to 4 is usually ideal for visualization
                    ax.set_ylim([-4, 4])
                
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.grid(True, alpha=0.1)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/Plot_{run_id}.png")
            plt.close() # Free memory

# 4. Final Save of Inertia Summary
inertia_df = pd.DataFrame(inertia_records)
inertia_df.to_csv(f"{results_dir}/Inertia_Summary_{run_id}.csv", index=False, sep=';')

print("Process finished. 60 iterations saved in data/clustering/")