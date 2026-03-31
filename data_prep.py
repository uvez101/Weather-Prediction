import pandas as pd
import requests

CITY = "New Damietta City"
print(f"🌍 Initiating API connection to fetch historical data for {CITY}...")

# 1. Open-Meteo Historical API URL
# Coordinates for New Damietta: Latitude 31.43, Longitude 31.68
# We request exactly what we need: Daily Max Temp, Mean Humidity, Max Wind Speed, and Precipitation
url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=31.43&longitude=31.68"
    "&start_date=2014-01-01&end_date=2024-01-01"
    "&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum"
    "&timezone=Africa%2FCairo"
)

# 2. Fetch the Data
response = requests.get(url)
data = response.json()

# 3. Convert the JSON response directly into a Pandas DataFrame
df = pd.DataFrame({
    'datetime': pd.to_datetime(data['daily']['time']),
    'temperature': data['daily']['temperature_2m_max'],     # Already the Daily High!
    'humidity': data['daily']['relative_humidity_2m_mean'],
    'wind_speed': data['daily']['wind_speed_10m_max'],
    'precipitation': data['daily']['precipitation_sum']     # We finally have precipitation!
})

# Set the date as the index
df.set_index('datetime', inplace=True)

# 4. Add our crucial Seasonality feature
df['month'] = df.index.month

# Clean any weird API glitches
df = df.dropna()

# 5. Build the 7-Day Forecasting "Sliding Window"
print("Building the 7-day future targets...")
for i in range(1, 8):
    df[f'target_day_{i}'] = df['temperature'].shift(-i)

# Drop the last 7 days since the future targets will be blank
df = df.dropna()

# 6. Save the final dataset
output_filename = "damietta_forecast_ready.csv"
df.to_csv(output_filename)

print(f"✅ Success! Clean Egyptian dataset saved as: {output_filename}")
print(df.head())