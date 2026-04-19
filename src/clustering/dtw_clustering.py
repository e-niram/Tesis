import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from itertools import product as iterproduct
from typing import Optional
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = 'results/clustering'
PLOTS_DIR   = os.path.join(BASE_DIR, 'plots')
METRICS_DIR = os.path.join(BASE_DIR, 'metrics')
TUNING_DIR  = os.path.join(BASE_DIR, 'tuning')
for _d in [PLOTS_DIR, METRICS_DIR, TUNING_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Static parameter grid ─────────────────────────────────────────────────────
N_CLUSTERS_LIST   = [2, 3, 4, 5]
MAX_ITER_LIST     = [20, 50]
N_INIT_LIST       = [3, 5]
TUNING_SEEDS      = [42, 43, 44]
PRODUCTION_SEED   = 42    # fixed for reproducibility
PRODUCTION_N_INIT = 10    # best of 10 random initializations

PERIODS       = ['nighttime', 'daytime']
PERIOD_LABELS = {'nighttime': 'Nocturno', 'daytime': 'Diurno'}

# Sakoe-Chiba radii expressed as fractions of the series length N.
# This makes the constraint scale with sampling resolution automatically.
#   5 %  → ~1 month of temporal drift on daily data
#   10 % → ~1 quarter of temporal drift
#   20 % → ~half a season of temporal drift
# Below 5 % the band is so tight it approximates Euclidean distance;
# above 25 % the warping is nearly unconstrained.
# Reference: Ratanamahatana & Keogh (2004).
SC_FRACTIONS = [0.05, 0.10, 0.20]

# ── Config helpers ────────────────────────────────────────────────────────────

def build_configs(n_timesteps: int) -> list[tuple]:
    """
    Return all (metric, metric_params) combinations for the given series length.
    Sakoe-Chiba radii are derived from n_timesteps so the band scales with the data.
    """
    configs = [('euclidean', None)]
    configs.append(('dtw', None))                   # unconstrained DTW
    for frac in SC_FRACTIONS:
        r = max(1, round(frac * n_timesteps))
        configs.append(('dtw', {'global_constraint': 'sakoe_chiba',
                                'sakoe_chiba_radius': r}))
    return configs


def config_label(metric: str, mp: Optional[dict]) -> str:
    """Human-readable (Spanish) label for a metric configuration."""
    if metric == 'euclidean':
        return 'Euclidiana'
    if mp is None:
        return 'DTW (sin restricción)'
    gc = mp.get('global_constraint', '')
    if gc == 'sakoe_chiba':
        return f"DTW Sakoe-Chiba r={mp['sakoe_chiba_radius']}"
    return f'DTW ({gc})'


def build_label_map(configs: list[tuple]) -> dict:
    """Map each config_label string back to its (metric, metric_params) tuple."""
    return {config_label(m, mp): (m, mp) for m, mp in configs}


def build_color_map(configs: list[tuple]) -> dict:
    """Assign a stable color to each config label."""
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    labels  = [config_label(m, mp) for m, mp in configs]
    return {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

# ── Silhouette helper ─────────────────────────────────────────────────────────

def compute_silhouette(X_2d: np.ndarray,
                       labels: np.ndarray,
                       metric: str,
                       dist_matrix: Optional[np.ndarray] = None) -> float:
    """
    Silhouette coefficient using the distance consistent with the clustering metric.
    - For euclidean: uses X_2d directly.
    - For DTW: requires a precomputed dist_matrix (pass it in to avoid
      recomputing the expensive cdist_dtw on every call).
    Returns NaN when the score is undefined (trivial solution or single cluster).
    """
    n_unique = len(set(labels))
    if n_unique < 2 or n_unique >= len(labels):
        return np.nan
    try:
        if metric == 'euclidean':
            return float(silhouette_score(X_2d, labels, metric='euclidean'))
        return float(silhouette_score(dist_matrix, labels, metric='precomputed'))
    except Exception:
        return np.nan


def build_dtw_dist_matrix(X_3d: np.ndarray, mp: Optional[dict]) -> np.ndarray:
    """
    Compute the pairwise DTW distance matrix for X_3d once per config.
    This is independent of k, n_init, max_iter, and seed, so it should be
    computed once and reused across all inner loop iterations.
    """
    kw: dict = {}
    if mp is not None and mp.get('global_constraint') == 'sakoe_chiba':
        kw = {'global_constraint': 'sakoe_chiba',
              'sakoe_chiba_radius': mp['sakoe_chiba_radius']}
    return cdist_dtw(X_3d, **kw)

# ── Tuning loop ───────────────────────────────────────────────────────────────
records = []

for period in PERIODS:
    print(f"{'='*60}\nPERIOD: {period}\n{'='*60}")

    df = pd.read_csv(f'data/test/{period}_final_test.csv', sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df_num = df.select_dtypes(include='number')

    X_2d: np.ndarray = df_num.T.values               # (n_stations, n_time)
    X_3d: np.ndarray = X_2d.reshape(*X_2d.shape, 1)  # (n_stations, n_time, 1)
    n_timesteps = X_2d.shape[1]

    configs     = build_configs(n_timesteps)
    color_map   = build_color_map(configs)

    n_fits = (len(configs) * len(MAX_ITER_LIST) * len(N_INIT_LIST) *
              len(N_CLUSTERS_LIST) * len(TUNING_SEEDS))
    print(f"  Series length: {n_timesteps} time steps")
    print(f"  SC radii:      {[round(f * n_timesteps) for f in SC_FRACTIONS]}"
          f"  (= {[f'{int(f*100)}%' for f in SC_FRACTIONS]} of N)")
    print(f"  Total fits:    {n_fits}\n")

    for metric, mp in configs:
        lbl = config_label(metric, mp)
        print(f"  Config: {lbl}")

        # Precompute the DTW distance matrix once per config — it depends only
        # on the data and the constraint, not on k, n_init, max_iter, or seed.
        dist_matrix = build_dtw_dist_matrix(X_3d, mp) if metric == 'dtw' else None

        for max_iter, n_init, k in iterproduct(MAX_ITER_LIST, N_INIT_LIST, N_CLUSTERS_LIST):
            inertias, sils = [], []

            for seed in TUNING_SEEDS:
                fit_kwargs = dict(
                    n_clusters=k,
                    metric=metric,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=seed,
                    n_jobs=-1,
                    dtw_inertia=(metric == 'dtw'),
                )
                if metric == 'dtw' and mp is not None:
                    fit_kwargs['metric_params'] = mp

                model  = TimeSeriesKMeans(**fit_kwargs)
                y_pred = model.fit_predict(X_3d)
                inertias.append(model.inertia_)
                sils.append(compute_silhouette(X_2d, y_pred, metric, dist_matrix))

            records.append(dict(
                period=period,
                metric=metric,
                metric_params=str(mp),
                config_label=lbl,
                max_iter=max_iter,
                n_init=n_init,
                n_clusters=k,
                mean_inertia=float(np.mean(inertias)),
                std_inertia=float(np.std(inertias)),
                mean_silhouette=float(np.nanmean(sils)),
                std_silhouette=float(np.nanstd(sils)),
            ))
            print(f"    k={k}  max_iter={max_iter}  n_init={n_init} → "
                  f"inertia={np.mean(inertias):.1f}±{np.std(inertias):.1f}  "
                  f"sil={np.nanmean(sils):.4f}")

tuning_df = pd.DataFrame(records)
tuning_df.to_csv(f"{METRICS_DIR}/Tuning_All.csv", index=False, sep=';')
print("\nTuning results saved to Tuning_All.csv")

# ── Plots ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def _mark_best(ax, xs, ys, color):
    """Highlight the point with the highest y value."""
    best_idx = int(np.argmax(ys))
    ax.scatter(xs[best_idx], ys[best_idx], s=80, zorder=5,
               color=color, edgecolors='black', linewidth=0.8)


for period in PERIODS:
    period_label = PERIOD_LABELS[period]
    pdf = tuning_df[tuning_df['period'] == period]

    # Rebuild the color map for this period using the same configs
    # (same fractions → same radii → same labels)
    sample_n = (pd.read_csv(f'data/test/{period}_final_test.csv', sep=';')
                  .select_dtypes(include='number').shape[0])
    period_configs    = build_configs(sample_n)
    period_color_map  = build_color_map(period_configs)
    period_label_map  = build_label_map(period_configs)

    # ── Plot A: metric / constraint comparison ────────────────────────────────
    # Average over max_iter and n_init; keep config_label and n_clusters
    agg = (pdf.groupby(['config_label', 'n_clusters'], as_index=False)
             .agg(
                 mean_inertia    =('mean_inertia',    'mean'),
                 mean_silhouette =('mean_silhouette', 'mean'),
             ))

    fig, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Selección de Configuración de Distancia — Período {period_label}',
        fontsize=13, fontweight='bold',
    )

    for lbl in agg['config_label'].unique():
        sub   = agg[agg['config_label'] == lbl].sort_values('n_clusters')
        color = period_color_map.get(lbl, 'grey')
        ks    = sub['n_clusters'].values
        mi    = sub['mean_inertia'].values
        ms    = sub['mean_silhouette'].values

        ax_elbow.plot(ks, mi, marker='o', label=lbl, color=color, linewidth=1.8)

        ax_sil.plot(ks, ms, marker='o', label=lbl, color=color, linewidth=1.8)
        _mark_best(ax_sil, ks, ms, color)

    ax_elbow.set_title('Método del Codo')
    ax_elbow.set_xlabel('Número de Clústeres')
    ax_elbow.set_ylabel('Inercia')
    ax_elbow.set_xticks(N_CLUSTERS_LIST)
    ax_elbow.legend()
    ax_elbow.grid(True, alpha=0.3)

    ax_sil.set_title('Coeficiente de Silueta')
    ax_sil.set_xlabel('Número de Clústeres')
    ax_sil.set_ylabel('Coeficiente de Silueta')
    ax_sil.set_xticks(N_CLUSTERS_LIST)
    ax_sil.legend()
    ax_sil.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{TUNING_DIR}/Seleccion_Configuracion_{period}.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # ── Plot B: sensitivity to max_iter and n_init ────────────────────────────
    best_label  = pdf.groupby('config_label')['mean_silhouette'].mean().idxmax()
    best_subset = pdf[pdf['config_label'] == best_label]

    combos = list(iterproduct(MAX_ITER_LIST, N_INIT_LIST))
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(combos)))

    fig, (ax_e, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Sensibilidad a Parámetros de Convergencia ({best_label}) — Período {period_label}',
        fontsize=12, fontweight='bold',
    )

    for idx, (max_iter, n_init) in enumerate(combos):
        sub = (best_subset[
                   (best_subset['max_iter'] == max_iter) &
                   (best_subset['n_init']   == n_init)
               ].sort_values('n_clusters'))
        line_label = f'max_iter={max_iter}, n_init={n_init}'
        c = colors[idx]
        ax_e.plot(sub['n_clusters'], sub['mean_inertia'],
                  marker='o', label=line_label, color=c, linewidth=1.8)
        ax_s.plot(sub['n_clusters'], sub['mean_silhouette'],
                  marker='o', label=line_label, color=c, linewidth=1.8)

    ax_e.set_title('Método del Codo')
    ax_e.set_xlabel('Número de Clústeres')
    ax_e.set_ylabel('Inercia')
    ax_e.set_xticks(N_CLUSTERS_LIST)
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)

    ax_s.set_title('Coeficiente de Silueta')
    ax_s.set_xlabel('Número de Clústeres')
    ax_s.set_ylabel('Coeficiente de Silueta')
    ax_s.set_xticks(N_CLUSTERS_LIST)
    ax_s.legend()
    ax_s.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{TUNING_DIR}/Sensibilidad_Convergencia_{period}.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

# ── Select best parameters per period ─────────────────────────────────────────
print(f"\n{'='*60}\nBest parameters selected by silhouette score\n{'='*60}")

best_per_period: dict = {}
for period in PERIODS:
    pdf  = tuning_df[tuning_df['period'] == period]
    idx  = pdf['mean_silhouette'].idxmax()
    best = pdf.loc[idx]
    best_per_period[period] = best
    print(f"\n  {period}:")
    print(f"    config_label : {best['config_label']}")
    print(f"    n_clusters   : {int(best['n_clusters'])}")
    print(f"    max_iter     : {int(best['max_iter'])}")
    print(f"    silhouette   : {best['mean_silhouette']:.4f}")
    print(f"    inertia      : {best['mean_inertia']:.2f}")

# ── Production run with best parameters ───────────────────────────────────────
# Single reproducible run per period: random_state fixes the sequence of random
# numbers; n_init=10 tries 10 different centroid initializations internally and
# keeps the best, so we get a robust solution without multiple outer seed loops.
print(f"\n{'='*60}\nProduction run "
      f"(seed={PRODUCTION_SEED}, n_init={PRODUCTION_N_INIT})\n{'='*60}")

inertia_records = []

for period in PERIODS:
    print(f"\n  Period: {period}")
    best = best_per_period[period]

    df = pd.read_csv(f'data/test/{period}_final_test.csv', sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    dates       = df['FECHA']
    df_num      = df.select_dtypes(include='number')
    station_ids = df_num.columns.tolist()
    X_2d = df_num.T.values
    X_3d = X_2d.reshape(*X_2d.shape, 1)

    label_map  = build_label_map(build_configs(X_2d.shape[1]))
    metric, mp = label_map[best['config_label']]
    k          = int(best['n_clusters'])
    max_iter   = int(best['max_iter'])

    run_id = f"{period}_k{k}"

    fit_kwargs = dict(
        n_clusters=k,
        metric=metric,
        max_iter=max_iter,
        n_init=PRODUCTION_N_INIT,
        random_state=PRODUCTION_SEED,
        n_jobs=-1,
        dtw_inertia=(metric == 'dtw'),
    )
    if metric == 'dtw' and mp is not None:
        fit_kwargs['metric_params'] = mp

    model  = TimeSeriesKMeans(**fit_kwargs)
    labels = model.fit_predict(X_3d)

    inertia_records.append({
        'Period': period, 'K': k,
        'Seed': PRODUCTION_SEED, 'NInit': PRODUCTION_N_INIT,
        'Inertia': model.inertia_,
    })

    (pd.DataFrame({'Station_ID': station_ids, 'Cluster': labels})
       .to_csv(f"{METRICS_DIR}/Results_{run_id}.csv", index=False, sep=';'))

    fig, axes = plt.subplots(k, 1, figsize=(10, 4 * k), sharex=True)
    if k == 1:
        axes = [axes]

    for i in range(k):
        ax             = axes[i]
        cluster_series = X_3d[labels == i, :, 0]
        for series in cluster_series:
            ax.plot(dates, series, color='grey', alpha=0.2)
        ax.plot(dates, model.cluster_centers_[i].ravel(), color='red', linewidth=2)
        ax.set_title(f'Clúster {i} — {len(cluster_series)} estaciones')
        ax.set_ylabel('Decibeles (dB)')
        ax.set_ylim([30, 105])
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.grid(True, alpha=0.1)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/Plot_{run_id}.png")
    plt.close()

    print(f"  inertia={model.inertia_:.1f}")

(pd.DataFrame(inertia_records)
   .to_csv(f"{METRICS_DIR}/Inertia_Summary.csv", index=False, sep=';'))

print("\nProcess finished.")
