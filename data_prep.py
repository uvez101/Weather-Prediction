import pandas as pd
import requests

CITY = "New Mansoura"
print(f"🌍 Initiating API connection to fetch comprehensive historical data for {CITY}...")

# 1. We added 'temperature_2m_min' to the API request
url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=31.43&longitude=31.68"
    "&start_date=2014-01-01&end_date=2024-01-01"
    "&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum"
    "&timezone=Africa%2FCairo"
)

# 2. Fetch the Data
response = requests.get(url)
data = response.json()

# 3. Build the core DataFrame
df = pd.DataFrame({
    'datetime': pd.to_datetime(data['daily']['time']),
    'temp_max': data['daily']['temperature_2m_max'],     
    'temp_min': data['daily']['temperature_2m_min'],     # <-- NEW Feature
    'humidity': data['daily']['relative_humidity_2m_mean'],
    'wind_speed': data['daily']['wind_speed_10m_max'],
    'precipitation': data['daily']['precipitation_sum']  
})

df.set_index('datetime', inplace=True)
df['month'] = df.index.month
df = df.dropna()

# 4. THE MULTI-TASK MATRIX (Generating 28 target columns!)
print("Building the 28-column 7-day future target matrix...")

# These are the 4 variables we want the AI to predict
target_variables = ['temp_max', 'temp_min', 'humidity', 'wind_speed']

for i in range(1, 8):
    for var in target_variables:
        # This will create columns like: temp_max_day_1, humidity_day_1, etc.
        df[f'{var}_day_{i}'] = df[var].shift(-i)

# Drop the last 7 days since the future targets will be blank
df = df.dropna()

# 5. Save the massive new dataset
output_filename = "damietta_forecast_ready.csv"
df.to_csv(output_filename)

print(f"✅ Success! Omni-dataset saved as: {output_filename}")
print(f"Total columns engineered: {len(df.columns)}")