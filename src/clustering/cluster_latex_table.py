import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from constants import STATIONS

MAPPING_FILES = {
    'nighttime': 'results/clustering/final/groupings/nighttime_DTW_Sakoe-Chiba_r5pct_k3_seed42.csv',
    'daytime':   'results/clustering/final/groupings/daytime_DTW_Sakoe-Chiba_r5pct_k3_seed42.csv',
}

DATE_CAP = '2025-12-31'

PERIOD_LABELS = {
    'daytime':   'Diurno',
    'nighttime': 'Nocturno',
}


def db_to_pressure(db):
    return 10 ** (db / 20)

def pressure_to_db(p):
    return 20 * np.log10(np.where(p > 0, p, 1e-12))

def energy_mean_db(series: pd.Series) -> float:
    return pressure_to_db(db_to_pressure(series.dropna()).mean())


def build_table(period: str, mapping_path: str) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_path, sep=';')
    mapping.columns = ['Station_ID', 'Cluster']
    mapping['Station_ID'] = mapping['Station_ID'].astype(int)

    df = pd.read_csv(f'data/final/{period}_final.csv', sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df.set_index('FECHA', inplace=True)
    df = df[df.index <= DATE_CAP]
    df.columns = df.columns.astype(int)

    # Per-station mean dB (energy average)
    station_means = {
        sid: energy_mean_db(df[sid])
        for sid in mapping['Station_ID']
        if sid in df.columns
    }

    # Cluster mean dB (energy average across all stations in cluster)
    cluster_means = {}
    cluster_sizes = {}
    for c in sorted(mapping['Cluster'].unique()):
        sids = mapping[mapping['Cluster'] == c]['Station_ID'].tolist()
        available = [s for s in sids if s in df.columns]
        cluster_sizes[c] = len(available)
        if available:
            mean_pressure = df[available].apply(db_to_pressure).mean(axis=1)
            cluster_means[c] = pressure_to_db(mean_pressure.values).mean()

    # Rank clusters low→high to assign noise label
    rank = sorted(cluster_means, key=cluster_means.get)
    noise_labels = {rank[0]: 'Bajo', rank[1]: 'Medio', rank[2]: 'Alto'}
    noise_order  = {'Alto': 0, 'Medio': 1, 'Bajo': 2}

    rows = []
    for _, row in mapping.sort_values(['Cluster', 'Station_ID']).iterrows():
        sid = int(row['Station_ID'])
        c   = int(row['Cluster'])
        rows.append({
            'Cluster':            c,
            'Ruido':              noise_labels[c],
            'ID Estación':        sid,
            'Estación':           STATIONS.get(sid, str(sid)),
            'Media Est. (dB)':    round(station_means.get(sid, float('nan')), 1),
            'Media Clúster (dB)': round(cluster_means.get(c, float('nan')), 1),
            'N Estaciones':       cluster_sizes.get(c, 0),
        })

    df = pd.DataFrame(rows)
    df['_order'] = df['Ruido'].map(noise_order)
    df = df.sort_values(['_order', 'ID Estación']).drop(columns='_order').reset_index(drop=True)
    return df


def df_to_latex(df: pd.DataFrame, period: str) -> str:
    label = PERIOD_LABELS[period]
    # columns: Ruido | N | Cluster mean | ID | Name | Station mean
    col_fmt = r'l c @{\hspace{1em}} r @{\hspace{1em}} r l @{\hspace{1.5em}} r'
    header  = r'Nivel de Ruido & N & Media Clúster (dB) & ID & Estación & Media Est. (dB)'

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\small',
        rf'\caption{{Asignación de estaciones por clúster — Período {label}}}',
        rf'\label{{tab:clusters_{period}}}',
        rf'\begin{{tabular}}{{{col_fmt}}}',
        r'\toprule',
        header + r' \\',
        r'\midrule',
    ]

    # Group rows by Ruido (already ordered Alto → Medio → Bajo)
    groups = df.groupby('Ruido', sort=False)
    ruido_order = df['Ruido'].unique()  # preserves insertion order after sort

    for i, ruido in enumerate(ruido_order):
        group = df[df['Ruido'] == ruido].reset_index(drop=True)
        n_rows = len(group)
        cluster_mean = group.iloc[0]['Media Clúster (dB)']
        n_stations   = group.iloc[0]['N Estaciones']

        for j, row in group.iterrows():
            is_first = (j == 0)
            ruido_cell   = rf'\multirow{{{n_rows}}}{{*}}{{{ruido}}}'   if is_first else ''
            mean_cl_cell = rf'\multirow{{{n_rows}}}{{*}}{{{cluster_mean:.1f}}}' if is_first else ''
            n_cell       = rf'\multirow{{{n_rows}}}{{*}}{{{n_stations}}}'       if is_first else ''

            cells = [
                ruido_cell,
                n_cell,
                mean_cl_cell,
                str(row['ID Estación']),
                row['Estación'],
                f"{row['Media Est. (dB)']:.1f}",
            ]
            lines.append(' & '.join(cells) + r' \\')

        if i < len(ruido_order) - 1:
            lines.append(r'\midrule')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'% Requires: \usepackage{booktabs,multirow}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


if __name__ == '__main__':
    for period, mapping_path in MAPPING_FILES.items():
        df = build_table(period, mapping_path)
        print(f'\n% ── {PERIOD_LABELS[period].upper()} ──────────────────────────────\n')
        print(df_to_latex(df, period))
