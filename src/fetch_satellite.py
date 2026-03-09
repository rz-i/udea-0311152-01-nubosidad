import pandas as pd
import requests
from datetime import datetime

def get_nasa_power_data(lat, lon, start_date, end_date):
    """
    Fetches hourly solar irradiance from NASA POWER API.
    """
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN", # All Sky Surface Shortwave Downward Irradiance
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date, # Formato YYYYMMDD
        "end": end_date,
        "format": "JSON"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extracting the time series
    records = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
    
    df_sat = pd.DataFrame(list(records.items()), columns=['timestamp_raw', 'sat_irradiance'])
    
    # NASA returns timestamp as YYYYMMDDHH (UTC)
    df_sat['timestamp'] = pd.to_datetime(df_sat['timestamp_raw'], format='%Y%m%d%H')
    
    # Convert to Local Time (Medellín is UTC-5)
    df_sat['timestamp'] = df_sat['timestamp'] - pd.Timedelta(hours=5)
    
    return df_sat[['timestamp', 'sat_irradiance']]

# --- Configuración ---
LAT, LON = 6.244, -75.581 # Medellín
START = "20260226" # Ajusta a tus fechas reales
END = "20260308"

print(f"[FETCH] Requesting data for Medellín from NASA...")
satellite_df = get_nasa_power_data(LAT, LON, START, END)
satellite_df.to_csv('data/satellite_data.csv', index=False)
print(f"[SUCCESS] Saved {len(satellite_df)} hourly satellite points to 'data/satellite_data.csv'")