import pandas as pd
import numpy as np

def audit_and_clean_gaps(file_path, threshold_cubic=14, threshold_seasonal=60, threshold_pct=20):
    """
    Audit station data quality and categorize them by gap severity.
    Now includes Max Gap Size and Percentage of Missing Values.
    """
    # 1. Load data
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df.set_index('FECHA', inplace=True)
    
    stations = [col for col in df.columns]
    total_days = len(df)
    
    # Store detailed station info
    station_stats = {}
    
    results = {
        'Cubic': [],     # Small gaps (use decifill cubic)
        'Seasonal': [],  # Medium gaps (use decifill seasonal)
        'Discard': []    # Critical gaps (too risky or > 20% missing)
    }

    print(f"--- Temporal Integrity Audit (TFM - Acoustic Data) ---")
    
    for col in stations:
        # Identify NaN blocks
        is_na = df[col].isna()
        missing_count = is_na.sum()
        missing_pct = (missing_count / total_days) * 100
        
        # Unique ID for each consecutive NaN block
        gap_groups = (is_na != is_na.shift()).cumsum()
        gap_lengths = is_na[is_na].groupby(gap_groups[is_na]).sum()
        
        max_gap = int(gap_lengths.max()) if not gap_lengths.empty else 0
        
        stats = {
            'id': col,
            'max_gap': max_gap,
            'missing_pct': round(missing_pct, 2)
        }
        
        # CATEGORIZATION LOGIC
        # Rule 1: If missing > 20%, discard regardless of gap size
        if missing_pct > threshold_pct:
            results['Discard'].append(stats)
        # Rule 2: Categorize based on max gap size
        elif max_gap <= threshold_cubic:
            results['Cubic'].append(stats)
        elif max_gap <= threshold_seasonal:
            results['Seasonal'].append(stats)
        else:
            results['Discard'].append(stats)

    # 2. Print Quality Report
    def print_group(name, stations_list, description):
        print(f"\n{'='*60}")
        print(f"GROUP: {name}")
        print(f"Criteria: {description}")
        print(f"{'='*60}")
        if not stations_list:
            print("No stations in this group.")
            return
        
        print(f"{'ID':<15} | {'Max Gap (Days)':<15} | {'Missing %':<12}")
        print("-" * 50)
        for s in stations_list:
            print(f"{s['id']:<15} | {s['max_gap']:<15} | {s['missing_pct']:<12}%")
        print(f"\nTotal: {len(stations_list)} stations")

    print_group("CUBIC", results['Cubic'], f"Max gap <= {threshold_cubic} days & Missing < {threshold_pct}%")
    print_group("SEASONAL", results['Seasonal'], f"{threshold_cubic} < Max gap <= {threshold_seasonal} days & Missing < {threshold_pct}%")
    print_group("DISCARD", results['Discard'], f"Max gap > {threshold_seasonal} days OR Missing > {threshold_pct}%")
    
    return results

# Execution
file_path = 'data/processed/LAeqNocturnoFiltrado.csv'
grupos = audit_and_clean_gaps(file_path)