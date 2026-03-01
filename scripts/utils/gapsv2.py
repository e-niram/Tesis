import pandas as pd
import numpy as np

def audit_and_clean_gaps(file_path, threshold_cubic=14, threshold_seasonal=60, threshold_pct=20):
    """
    Audit station data quality and categorize them by gap severity.
    Includes a full list of all gap lengths found in each station.
    """
    # 1. Load data
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df.set_index('FECHA', inplace=True)
    
    stations = [col for col in df.columns]
    total_days = len(df)
    
    results = {
        'Cubic': [],     
        'Seasonal': [],  
        'Discard': []    
    }

    print(f"--- Comprehensive Gap Audit (TFM - Acoustic Data) ---")
    
    for col in stations:
        is_na = df[col].isna()
        missing_count = is_na.sum()
        missing_pct = (missing_count / total_days) * 100
        
        # Identify all gap lengths
        gap_groups = (is_na != is_na.shift()).cumsum()
        # This creates a list of lengths for every NaN block found
        all_gaps = is_na[is_na].groupby(gap_groups[is_na]).sum().astype(int).tolist()
        
        max_gap = max(all_gaps) if all_gaps else 0
        
        stats = {
            'id': col,
            'max_gap': max_gap,
            'all_gaps': all_gaps,
            'missing_pct': round(missing_pct, 2)
        }
        
        # Categorization Logic
        if missing_pct > threshold_pct:
            results['Discard'].append(stats)
        elif max_gap <= threshold_cubic:
            results['Cubic'].append(stats)
        elif max_gap <= threshold_seasonal:
            results['Seasonal'].append(stats)
        else:
            results['Discard'].append(stats)

    # 2. Print Detailed Quality Report
    def print_group(name, stations_list, description):
        print(f"\n{'='*80}")
        print(f"GROUP: {name}")
        print(f"Criteria: {description}")
        print(f"{'='*80}")
        if not stations_list:
            print("No stations in this group.")
            return
        
        # Header with All Gaps column
        print(f"{'ID':<12} | {'Max Gap':<8} | {'Missing %':<10} | {'All Gap Lengths (Days)'}")
        print("-" * 80)
        for s in stations_list:
            gaps_str = ", ".join(map(str, sorted(s['all_gaps'], reverse=True)))
            print(f"{s['id']:<12} | {s['max_gap']:<8} | {s['missing_pct']:<9}% | [{gaps_str}]")
        print(f"\nTotal: {len(stations_list)} stations")

    print_group("CUBIC", results['Cubic'], f"Max gap <= {threshold_cubic}d & Missing < {threshold_pct}%")
    print_group("SEASONAL", results['Seasonal'], f"{threshold_cubic}d < Max gap <= {threshold_seasonal}d & Missing < {threshold_pct}%")
    print_group("DISCARD", results['Discard'], f"Max gap > {threshold_seasonal}d OR Missing > {threshold_pct}%")
    
    return results

# Execution
file_path = 'data/processed/LAeqNocturnoFiltrado.csv'
grupos = audit_and_clean_gaps(file_path)