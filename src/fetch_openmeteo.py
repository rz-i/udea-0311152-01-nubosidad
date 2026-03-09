import pandas as pd
import requests

def get_openmeteo_data(lat, lon, start_date, end_date):
    """
    Fetches hourly solar radiation from Open-Meteo (ERA5/Historical).
    Dates must be in YYYY-MM-DD format.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation",
        "timezone": "UTC"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Extracting the time series
    hourly = data['hourly']
    df_sat = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time']),
        'sat_radiance': hourly['shortwave_radiation']
    })
    
    return df_sat

# Configuración (Usa tus fechas reales en formato YYYY-MM-DD)
LAT, LON = 6.244, -75.581
START, END = "2026-02-26", "2026-03-09"

print(f"[FETCH] Requesting data from Open-Meteo (UTC)...")
satellite_df = get_openmeteo_data(LAT, LON, START, END)

# Filter out night data (Radiance = 0) to make the merge cleaner
satellite_df = satellite_df[satellite_df['sat_radiance'] > 0]

satellite_df.to_csv('data/satellite_data.csv', index=False)
print(f"[SUCCESS] Saved {len(satellite_df)} daylight points to 'data/satellite_data.csv'")