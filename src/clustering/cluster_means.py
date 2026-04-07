import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Fill in the mapping file for each period
MAPPING_FILES = {
    'nighttime': 'results/clustering/metrics/Results_nighttime_k3_s40.csv',
    'daytime':   'results/clustering/metrics/Results_daytime_k3_s40.csv',
}

NOISE_LABELS = {
    'low':    ('green',  'Ruido bajo'),
    'medium': ('yellow', 'Ruido medio'),
    'high':   ('red',    'Ruido alto'),
}

def db_to_pressure(db):
    return 10**(db / 20)

def pressure_to_db(p):
    return 20 * np.log10(np.where(p > 0, p, 1e-12))

def calculate_cluster_means(period, mapping_path):
    data_path  = f'data/final/{period}_final.csv'
    output_csv = f'results/clustering/metrics/Cluster_Means_{period}.csv'
    img_individual  = f'results/clustering/plots/cluster_{period}_profiles_individuales.png'
    img_comparativo = f'results/clustering/plots/cluster_{period}_profiles_superpuesto.png'

    # Load data
    mapping = pd.read_csv(mapping_path, sep=';')
    mapping.columns = ['Station_ID', 'Cluster']

    df_data = pd.read_csv(data_path, sep=';')
    df_data['FECHA'] = pd.to_datetime(df_data['FECHA'])
    df_data.set_index('FECHA', inplace=True)

    # Calculate cluster means
    clusters = sorted(mapping['Cluster'].unique())
    cluster_means_df = pd.DataFrame(index=df_data.index)

    for c in clusters:
        stations = mapping[mapping['Cluster'] == c]['Station_ID'].astype(str).tolist()
        available = [s for s in stations if s in df_data.columns]
        if available:
            mean_pressure = df_data[available].apply(db_to_pressure).mean(axis=1)
            cluster_means_df[f'Cluster_{c}'] = pressure_to_db(mean_pressure)
            print(f"[{period}] Cluster {c}: {len(available)} estaciones.")

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cluster_means_df.to_csv(output_csv, sep=';')

    # Assign colors by overall mean (lowest→green, middle→yellow, highest→red)
    overall_means = {c: cluster_means_df[f'Cluster_{c}'].mean()
                     for c in clusters if f'Cluster_{c}' in cluster_means_df.columns}
    rank = sorted(overall_means, key=overall_means.get)  # ascending
    color_map  = {rank[0]: 'green', rank[1]: '#CC8400', rank[2]: 'red'}
    label_map  = {rank[0]: 'Ruido bajo', rank[1]: 'Ruido medio', rank[2]: 'Ruido alto'}

    os.makedirs(os.path.dirname(img_individual), exist_ok=True)

    # Image 1: individual subplots
    fig1, axes = plt.subplots(nrows=len(clusters), ncols=1, figsize=(12, 15), sharex=True)

    for i, c in enumerate(clusters):
        col = f'Cluster_{c}'
        if col in cluster_means_df.columns:
            ax = axes[i]
            ax.plot(cluster_means_df.index, cluster_means_df[col],
                    color=color_map[c], lw=1, label=label_map[c])
            ax.set_ylim(35, 80)
            ax.set_title(f'Perfil Medio del Cluster {c} ({label_map[c]})', fontsize=14, fontweight='bold')
            ax.set_ylabel('Nivel Sonoro (dB)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

    fig1.tight_layout()
    fig1.savefig(img_individual, dpi=300)
    plt.close(fig1)
    print(f"[{period}] Imagen individual guardada: {img_individual}")

    # Image 2: overlaid plot
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    for c in clusters:
        col = f'Cluster_{c}'
        if col in cluster_means_df.columns:
            ax2.plot(cluster_means_df.index, cluster_means_df[col],
                     color=color_map[c], lw=1.5, alpha=0.8, label=label_map[c])

    ax2.set_ylim(35, 80)
    period_label = 'Nocturno' if period == 'nighttime' else 'Diurno'
    ax2.set_title(f'Comparativa de Perfiles Medios por Cluster (Ruido {period_label})', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Nivel Sonoro (dB)')
    ax2.set_xlabel('Fecha')
    ax2.grid(True, linestyle='-', alpha=0.3)
    ax2.legend(loc='upper right', ncol=3)

    fig2.tight_layout()
    fig2.savefig(img_comparativo, dpi=300)
    plt.close(fig2)
    print(f"[{period}] Imagen comparativa guardada: {img_comparativo}")

if __name__ == "__main__":
    for period, mapping_path in MAPPING_FILES.items():
        calculate_cluster_means(period, mapping_path)
