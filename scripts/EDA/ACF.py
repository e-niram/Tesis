import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

def analyze_weekly_seasonality(file_path, column_name, lags=35):
    # 1. Cargar y preparar
    df = pd.read_csv(file_path, sep=';')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df.set_index('FECHA', inplace=True)
    
    # Limpiar NaNs (ACF no los soporta)
    series = df[column_name].dropna()
    
    # 2. Calcular valores de ACF para reporte técnico
    acf_values = acf(series, nlags=lags)
    day_7_acf = acf_values[7]
    
    print(f"--- Análisis de Autocorrelación para {column_name} ---")
    print(f"Coeficiente de autocorrelación en Lag 7 (1 semana): {day_7_acf:.4f}")
    
    if day_7_acf > 0.2: # Umbral común para estacionalidad relevante
        print("Resultado: Se confirma una estacionalidad semanal FUERTE.")
    else:
        print("Resultado: Estacionalidad semanal débil o inexistente.")

    # 3. Visualización profesional
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ACF
    plot_acf(series, lags=lags, ax=ax1, color='teal', alpha=0.05)
    ax1.set_title(f'ACF - Plaza del Carmen (Diurno)')
    ax1.set_xlabel('Retardos (Días)')
    # Resaltar múltiplos de 7
    for day in range(7, lags + 1, 7):
        ax1.axvline(x=day, color='orange', linestyle='--', alpha=0.5, label='Ciclo Semanal' if day==7 else "")

    # PACF
    plot_pacf(series, lags=lags, ax=ax2, color='darkblue', alpha=0.05, method='ywm')
    ax2.set_title(f'PACF - Plaza del Carmen (Diurno)')
    ax2.set_xlabel('Retardos (Días)')

    plt.tight_layout()
    plt.show()

# Ejecución (ajusta a tu archivo)
file_path = 'data/processed/LAeqDiurno.csv'
analyze_weekly_seasonality(file_path, '3')