import pandas as pd

CITY = 'Seattle' 
print(f"🔧 Starting data prep for {CITY}...")

# 1. Load the raw files (Make sure these are in your folder!)
print("Loading raw CSVs (this might take a few seconds)...")
temp = pd.read_csv('temperature.csv', parse_dates=['datetime'], index_col='datetime')
humidity = pd.read_csv('humidity.csv', parse_dates=['datetime'], index_col='datetime')
wind = pd.read_csv('wind_speed.csv', parse_dates=['datetime'], index_col='datetime')
# Note: we are skipping precipitation since this Kaggle dataset doesn't have it

# 2. Extract just our target city
df = pd.DataFrame({
    'temperature': temp[CITY],
    'humidity': humidity[CITY],
    'wind_speed': wind[CITY]
})

# 3. Convert Kelvin to Celsius
df['temperature'] = df['temperature'] - 273.15

# 4. Resample Hourly Data to Daily (taking the daily average)
print("Converting hourly data to daily averages...")
df_daily = df.resample('D').mean()

df_daily['month'] = df_daily.index.month

# 5. Create the 7-Day Forecasting "Sliding Window"
print("Building the 7-day future targets...")
for i in range(1, 8):
    df_daily[f'target_day_{i}'] = df_daily['temperature'].shift(-i)

# 6. Clean up missing data (NaNs)
# This drops the last 7 days (since we can't know the future yet) 
# and any days where sensors were broken.
df_daily = df_daily.dropna()

# 7. Save the final, clean dataset!
output_filename = f"{CITY.lower()}_forecast_ready.csv"
df_daily.to_csv(output_filename)

print(f"✅ Success! Clean dataset saved as: {output_filename}")
print(df_daily.head())