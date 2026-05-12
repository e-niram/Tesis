import os
import pandas as pd

os.chdir(os.path.join(os.path.dirname(__file__), '../..'))

from dtw_clustering import plot_config_comparison, CHECKPOINT_FILE, PERIODS

tuning_df = pd.read_csv(CHECKPOINT_FILE, sep=';')
for period in PERIODS:
    plot_config_comparison(tuning_df, period, exclude_labels=['Euclidiana'])

