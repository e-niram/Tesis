import pandas as pd
import os

def process_and_synchronize_noise_data():
    # 1. Configuration
    file_path = 'data/processed/LAeqDiurno.csv'
    output_path = 'data/processed/LAeqDiurnoFiltrado.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 2. Load the dataset
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    
    # 3. FIX TIME GAPS (Dates that are completely missing from the CSV)
    # We create a reference range from the very first to the very last date present
    full_range = pd.date_range(start=df['FECHA'].min(), end=df['FECHA'].max(), freq='D')
    
    df.set_index('FECHA', inplace=True)
    
    # Reindexing inserts rows for missing dates as NaN
    df_reindexed = df.reindex(full_range)
    df_reindexed.index.name = 'FECHA'
    df_reindexed.reset_index(inplace=True)
    
    missing_dates_count = len(full_range) - len(df)
    print(f"Time Gap Analysis: {missing_dates_count} missing dates were inserted into the timeline.")

    # 4. IDENTIFY FIRST COMPLETE RECORD (Integrity Filter)
    # Now that the timeline is continuous, we look for the first row with 0 NAs
    # We exclude the 'FECHA' column from the NA check
    stations_columns = [col for col in df_reindexed.columns if col != 'FECHA']
    complete_cases = df_reindexed.dropna(subset=stations_columns)

    if not complete_cases.empty:
        # Get the index of the first record where all stations have data
        first_complete_idx = complete_cases.index[0]
        first_complete_date = df_reindexed.loc[first_complete_idx, 'FECHA'].strftime('%Y-%m-%d')
        
        print(f"Integrity Filter: Starting sequence from {first_complete_date}")

        # 5. SLICE DATAFRAME
        # We keep only rows from the first complete date onwards
        df_final = df_reindexed.iloc[first_complete_idx:].copy()

        # 6. SAVE CLEANED DOCUMENT
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, sep=';', index=False)
        
        print(f"Final Report:")
        print(f" - Rows removed before first complete date: {first_complete_idx}")
        print(f" - Final dataset shape: {df_final.shape}")
        print(f" - Cleaned file saved to: {output_path}")
    else:
        print("Operation aborted: No rows found with complete data for all stations.")

if __name__ == "__main__":
    process_and_synchronize_noise_data()