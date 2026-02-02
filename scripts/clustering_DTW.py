from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import os
import pandas as pd

def prepare_data_for_clustering(df):
    """
    Converts the processed noise data into a long format suitable for tslearn.
    Uses NMT as 'building_id' and FECHA as 'timestamp'.
    """
    # We select only the columns needed for clustering
    # Using Daytime noise (LAeqDiurno) as the feature
    data_long = (
        df.reset_index()
        .loc[:, ['FECHA', 'NMT', 'LAeqDiurno']]
        .rename(columns={'FECHA': 'timestamp', 'NMT': 'station_id', 'LAeqDiurno': 'noise_level'})
        .set_index(['station_id', 'timestamp'])
        .sort_index(ascending=True)
    )
    return data_long

def run_dtw_clustering(data_long, n_clusters=4):
    """
    Fits a TimeSeriesKMeans model using the DTW metric.
    """
    # Scaling is crucial for DTW to compare shapes rather than absolute magnitudes
    scaler = TimeSeriesScalerMeanVariance()
    
    # TimeSeriesKMeans expects a specific shape (n_series, n_timestamps, n_features)
    # We pivot to ensure all series are aligned in time
    data_pivoted = data_long.unstack(level=0).T # station_id becomes rows
    formatted_series = scaler.fit_transform(data_pivoted.values)

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="euclidean",
        max_iter=10,
        random_state=123456,
        n_jobs=-1 # Use all cores
    )

    clusters_idx = model.fit_predict(formatted_series)
    
    # Create results dataframe
    clusters_df = pd.DataFrame({
        'station_id': data_pivoted.index.get_level_values('station_id'),
        'cluster_based_on_dtw': clusters_idx.astype(str)
    })
    
    return model, clusters_df

def merge_clusters_to_main(df, clusters_df):
    """
    Merges cluster assignments back into the original noise dataframe.
    """
    # Ensure ID types match (integers)
    df['NMT'] = df['NMT'].astype(int)
    clusters_df['station_id'] = clusters_df['station_id'].astype(int)

    df_final = pd.merge(
        df.reset_index(),
        clusters_df,
        left_on='NMT',
        right_on='station_id',
        how='left',
        validate='m:1'
    ).drop(columns=['station_id'])
    
    return df_final

def save_dtw_results(df, filename="ruido_dtw.csv"):
   """
   Saves the final dataframe with cluster assignments to the processed data folder.
   """
   output_dir = "data/processed"
   
   # Ensure directory exists
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      
   path = os.path.join(output_dir, filename)
   
   # Save using semicolon as separator to maintain consistency with previous files
   df.to_csv(path, index=False, sep=';')
   print(f"DTW results successfully saved to: {path}")

def run_dtw_pipeline_2025():
    # 1. Load Data
    file_path = "data/processed/ruido_processed.csv"
    df = pd.read_csv(file_path, sep=';')
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    
    # 2. Filter from January 1st, 2025 onwards
    # This ensures we only cluster recent noise patterns
    df_2025 = df[df["FECHA"] >= "2025-01-01"].copy()
    
    if df_2025.empty:
        print("Error: No data found from 2025-01-01 onwards.")
        return

    print("Revision NAs = ", df.isna().sum())
    # 3. Prepare Long Format
    data_long = (
        df_2025.loc[:, ['FECHA', 'NMT', 'LAeqDiurno']]
        .rename(columns={'FECHA': 'timestamp', 'NMT': 'station_id', 'LAeqDiurno': 'noise_level'})
        .set_index(['station_id', 'timestamp'])
        .sort_index(ascending=True)
    )

    # 4. Fit DTW Clustering
    # We pivot to align stations by time for the scaler
    data_pivoted = data_long.unstack(level=0).T 
    
    # Scaling ensures we compare the 'shape' of the noise (rhythm) 
    # rather than just the volume level
    scaler = TimeSeriesScalerMeanVariance()
    formatted_series = scaler.fit_transform(data_pivoted.values)

    model = TimeSeriesKMeans(
        n_clusters=4,
        metric="dtw",
        max_iter=10,
        random_state=123456,
        n_jobs=-1
    )

    clusters_idx = model.fit_predict(formatted_series)
    
    # 5. Prepare Results
    clusters_df = pd.DataFrame({
        'NMT': data_pivoted.index.get_level_values('station_id').astype(int),
        'cluster_based_on_dtw_2025': clusters_idx.astype(str)
    })

    # 6. Merge and Save
    df_final = pd.merge(
        df_2025, 
        clusters_df, 
        on='NMT', 
        how='left'
    )
    
    output_path = "data/processed/ruido_dtw_2025.csv"
    df_final.to_csv(output_path, index=False, sep=';')
    print(f"2025 DTW clustering complete. Saved to: {output_path}")

run_dtw_pipeline_2025()