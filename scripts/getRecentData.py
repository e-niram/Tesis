import pandas as pd

# 1. Load Data
file_path = 'data/processed/DiurnoImputado.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 2. Convert FECHA to datetime objects
# It is vital to ensure the column is in the correct format for logical comparisons
df['FECHA'] = pd.to_datetime(df['FECHA'])

# 3. Apply Temporal Filter
# We define the threshold as December 1st, 2023
cutoff_date = '2023-12-01'

# We use a boolean mask to keep only observations from that date onwards
# Using .copy() avoids the SettingWithCopyWarning in future manipulations
df_filtered = df[df['FECHA'] >= cutoff_date].copy()

# 4. Technical Validation & Logging
print(f"--- Data Filtering Report ---")
print(f"Original observations: {len(df)}")
print(f"Observations after filter: {len(df_filtered)}")

if not df_filtered.empty:
    min_date = df_filtered['FECHA'].min().strftime('%Y-%m-%d')
    max_date = df_filtered['FECHA'].max().strftime('%Y-%m-%d')
    print(f"Date range in new dataset: {min_date} to {max_date}")
else:
    print("Warning: The resulting dataset is empty. Check the cutoff date.")

# 5. Export Processed Data (Optional)
df_filtered.to_excel('data/processed/DiurnoReciente.xlsx', index=False)

# The 'df_filtered' variable is now ready for clustering or mapping tasks