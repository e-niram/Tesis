# Tesis

Acoustic noise analysis for Madrid's 31 monitoring stations using DTW-based time series clustering (TFM).

## Project Structure

```
├── src/
│   ├── constants.py          # Station IDs, names, and GPS coordinates
│   ├── pipeline/             # Sequential data processing steps
│   │   ├── 01_process_raw.py     # Clean and reshape raw noise data
│   │   ├── 02_split_periods.py   # Split into daytime/nighttime series
│   │   └── 03_handle_missing.py  # Synchronize timeline and impute gaps
│   ├── analysis/             # Exploratory analysis tools
│   │   ├── acf.py, eda.py, eda_recent.py
│   │   ├── fetch_recent_data.py
│   │   └── filter_stations.py
│   ├── clustering/           # DTW K-means clustering
│   │   ├── dtw_clustering.py     # Main clustering loop (multi-seed)
│   │   ├── cluster_means.py      # Compute cluster centroid profiles
│   │   ├── dtw_plot.py           # DTW alignment visualization
│   │   └── station_map.py        # Geospatial cluster map
│   └── utils/
│       ├── gaps.py               # Gap audit and categorization
│       └── viability.py          # Station data viability checks
├── data/
│   ├── raw/                  # Original source data (do not modify)
│   ├── processed/            # Intermediate pipeline outputs
│   ├── final/                # Final imputed datasets for clustering
│   └── test/                 # Sample data for validation
├── notebooks/
│   └── cleaning.ipynb        # Exploratory data cleaning notebook
├── app/
│   └── app.py                # Streamlit forecast dashboard
├── sas/                      # Supplementary SAS analysis
└── results/
    └── clustering/           # Plots and metrics from clustering runs
        ├── plots/
        └── metrics/
```

## Pipeline

```
data/raw/ → 01_process_raw.py → data/processed/noise_processed.csv
                              → 02_split_periods.py → daytime.csv / nighttime.csv
                              → 03_handle_missing.py → data/final/*_final.csv
                              → dtw_clustering.py → results/clustering/
```

## How to Run

**Pipeline (run in order):**
```bash
python src/pipeline/01_process_raw.py
python src/pipeline/02_split_periods.py
python src/pipeline/03_handle_missing.py
```

**Clustering:**
```bash
caffeinate -i python src/clustering/dtw_clustering.py
```

**Dashboard:**
```bash
streamlit run app/app.py
```
