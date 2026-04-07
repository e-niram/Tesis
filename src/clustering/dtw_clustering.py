import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from tslearn.clustering import TimeSeriesKMeans

# 1. Setup paths and directories
base_dir = 'results/clustering'
plots_dir = os.path.join(base_dir, 'plots')
results_dir = os.path.join(base_dir, 'metrics')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

n_clusters_list = [2, 3, 4]
seeds = range(40, 55)  # 15 different seeds

for PERIOD in ["nighttime", "daytime"]:
    print(f"Processing period: {PERIOD}...")

    # 2. Load and Prepare Data
    file_path = f'data/final/{PERIOD}_final.csv'
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    dates = df['FECHA']

    df_numeric = df.select_dtypes(include=['number'])
    station_ids = df_numeric.columns
    X_orig = df_numeric.T.values

    # List for capturing inertia results for this period
    inertia_records = []

    # 3. Execution Loop
    for n_clusters in n_clusters_list:
        for seed in seeds:
            run_id = f"{PERIOD}_k{n_clusters}_s{seed}"

            # Initialize and fit model
            model = TimeSeriesKMeans(
                n_clusters=n_clusters,
                metric="dtw",
                max_iter=20,
                random_state=seed,
                n_jobs=-1,
                dtw_inertia=True
            )

            cluster_labels = model.fit_predict(X_orig)
            current_inertia = model.inertia_

            # A. Save Inertia to list
            inertia_records.append({
                'K': n_clusters,
                'Seed': seed,
                'Inertia': current_inertia
            })

            # B. Save Station Results
            results = pd.DataFrame({'Station_ID': station_ids, 'Cluster': cluster_labels})
            results.to_csv(f"{results_dir}/Results_{run_id}.csv", index=False, sep=';')

            # C. Generate and Save Plot
            fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 4 * n_clusters), sharex=True)
            if n_clusters == 1:
                axes = [axes]

            for i in range(n_clusters):
                ax = axes[i]
                cluster_series = X_orig[cluster_labels == i]

                for series in cluster_series:
                    ax.plot(dates, series.ravel(), color='grey', alpha=0.2)

                ax.plot(dates, model.cluster_centers_[i].ravel(), color='red', linewidth=2)
                ax.set_title(f'Clúster {i} - Cantidad de Estaciones: {len(cluster_series)}')
                ax.set_ylabel('Decibeles (dB)')
                ax.set_ylim([30, 105])
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                ax.grid(True, alpha=0.1)

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/Plot_{run_id}.png")
            plt.close()

    # 4. Save Inertia Summary for this period
    inertia_df = pd.DataFrame(inertia_records)
    inertia_df.to_csv(f"{results_dir}/Inertia_Summary_{PERIOD}.csv", index=False, sep=';')
    print(f"  Done. Results saved for {PERIOD}.")

print("Process finished.")
