import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Madrid Noise Forecast", page_icon="🔊", layout="wide")

st.title("🔊 Madrid Noise Level (LAeq) Prediction Map")
st.markdown("""
Interactive dashboard displaying historical and forecasted acoustic pollution across Madrid's 31 monitoring stations.
""")

# --- 2. DATA GENERATION (MOCK ML MODEL OUTPUT) ---
@st.cache_data
def load_station_data():
    """Simulates 31 stations around Madrid's coordinates."""
    np.random.seed(42)
    # Madrid rough bounding box
    lats = np.random.uniform(40.35, 40.50, 31)
    lons = np.random.uniform(-3.75, -3.60, 31)
    stations = [f"Station_{i+1:02d}" for i in range(31)]
    
    return pd.DataFrame({"Station": stations, "Lat": lats, "Lon": lons})

@st.cache_data
def load_time_series_data(forecast_days=7):
    """Simulates historical data and time-series predictions with confidence intervals."""
    stations = load_station_data()["Station"].tolist()
    dates_hist = [datetime.now().date() - timedelta(days=i) for i in range(30, 0, -1)]
    dates_fore = [datetime.now().date() + timedelta(days=i) for i in range(1, 31)] # Max 30 days forecast
    
    data = []
    
    for station in stations:
        # Base noise level for the station (e.g., some are near highways, some in parks)
        base_noise = np.random.uniform(45, 75) 
        
        # Historical
        for d in dates_hist:
            noise = base_noise + np.random.normal(0, 3)
            data.append({"Station": station, "Date": d, "Type": "Historical", "LAeq": noise, "Lower_CI": np.nan, "Upper_CI": np.nan})
            
        # Forecast
        for i, d in enumerate(dates_fore):
            # Add trend and increasing uncertainty
            noise = base_noise + np.sin(i) * 2 + np.random.normal(0, 1)
            uncertainty = 1 + (i * 0.1) # Confidence interval widens over time
            data.append({
                "Station": station, 
                "Date": d, 
                "Type": "Forecast", 
                "LAeq": noise, 
                "Lower_CI": noise - uncertainty, 
                "Upper_CI": noise + uncertainty
            })
            
    return pd.DataFrame(data)

# Load base data
df_stations = load_station_data()
df_ts = load_time_series_data()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Filter & Controls")

# Forecast Range Slider
forecast_range = st.sidebar.slider("Forecast Range (Days)", min_value=1, max_value=30, value=7)

# Station Selector
selected_station = st.sidebar.selectbox("Select Station for Deep Dive", ["All Stations"] + df_stations["Station"].tolist())

# --- 4. DATA FILTERING ---
# Filter time series by selected forecast range
df_ts['Date'] = pd.to_datetime(df_ts['Date'])
max_date = datetime.now() + timedelta(days=forecast_range)
df_filtered = df_ts[df_ts['Date'] <= max_date]

# Get the most recent forecast value to color the map
df_latest_forecast = df_filtered[df_filtered['Type'] == 'Forecast'].groupby('Station').first().reset_index()
df_map = pd.merge(df_stations, df_latest_forecast[['Station', 'LAeq']], on='Station')

# Assign Severity Colors based on LAeq (WHO inspired)
def get_severity_color(laeq):
    if laeq < 55: return "green"
    elif laeq < 65: return "orange"
    else: return "red"

df_map['Color'] = df_map['LAeq'].apply(get_severity_color)
df_map['Size'] = df_map['LAeq'] # Adjust marker size by noise

# --- 5. MAIN LAYOUT: MAP ---
st.subheader("🗺️ Spatial Distribution (Next 24h Forecast)")

fig_map = px.scatter_mapbox(
    df_map, 
    lat="Lat", 
    lon="Lon", 
    hover_name="Station", 
    hover_data={"LAeq": ":.1f", "Lat": False, "Lon": False, "Color": False, "Size": False},
    color="Color",
    size="Size",
    color_discrete_map="identity",
    zoom=10.5, 
    height=500,
    title="Madrid Acoustic Sensors"
)
fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# --- 6. MAIN LAYOUT: TIME SERIES PREDICTION ---
st.subheader(f"📈 Time Series Forecast: {selected_station}")

if selected_station == "All Stations":
    st.info("Select a specific station from the sidebar to view its detailed forecast and confidence intervals.")
else:
    station_data = df_filtered[df_filtered['Station'] == selected_station]
    hist_data = station_data[station_data['Type'] == 'Historical']
    fore_data = station_data[station_data['Type'] == 'Forecast']

    fig_ts = go.Figure()

    # Historical Trace
    fig_ts.add_trace(go.Scatter(
        x=hist_data['Date'], y=hist_data['LAeq'],
        mode='lines+markers', name='Historical Data',
        line=dict(color='blue')
    ))

    # Forecast Trace
    fig_ts.add_trace(go.Scatter(
        x=fore_data['Date'], y=fore_data['LAeq'],
        mode='lines+markers', name='Model Prediction',
        line=dict(color='red', dash='dash')
    ))

    # Confidence Interval (Shaded Area)
    fig_ts.add_trace(go.Scatter(
        x=pd.concat([fore_data['Date'], fore_data['Date'][::-1]]),
        y=pd.concat([fore_data['Upper_CI'], fore_data['Lower_CI'][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Confidence Interval (95%)'
    ))

    fig_ts.update_layout(
        xaxis_title="Date",
        yaxis_title="Noise Level (LAeq dB)",
        hovermode="x unified",
        shapes=[
            # Add a vertical line for "Today"
            dict(type="line", x0=datetime.now().date(), y0=0, x1=datetime.now().date(), y1=1,
                 xref="x", yref="paper", line=dict(color="Black", width=2, dash="dot"))
        ]
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)