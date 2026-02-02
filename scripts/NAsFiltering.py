import pandas as pd
import os

# 1. Load the dataset
file_path = 'data/processed/LAeqDiurno.csv'
df = pd.read_csv(file_path, sep=';')

# 2. Identify rows with zero missing values
complete_cases = df.dropna()

if not complete_cases.empty:
    # Get the index of the first record where all stations have data
    first_complete_idx = complete_cases.index[0]
    first_complete_date = df.loc[first_complete_idx, 'FECHA']
    
    print(f"Filtering data starting from: {first_complete_date}")

    # 3. Slice the dataframe to remove all rows before first_complete_date
    # .iloc[first_complete_idx:] includes the first complete row and everything after it
    df_filtered = df.iloc[first_complete_idx:].copy()

    # 4. Save the cleaned document
    output_path = 'data/processed/LAeqDiurnoFiltrado.csv'
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with semicolon separator to maintain consistency
    df_filtered.to_csv(output_path, sep=';', index=False)
    
    print(f"Cleaned file saved successfully to: {output_path}")
    print(f"Removed {first_complete_idx} leading rows with missing data.")
else:
    print("Operation aborted: No rows found with complete data for all stations.")