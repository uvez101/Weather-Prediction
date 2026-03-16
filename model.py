import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("🧠 Booting up the training sequence...")

# 1. Load the pristine dataset from our ETL pipeline
# We set index_col='datetime' so Pandas knows the dates aren't features
df = pd.read_csv('seattle_forecast_ready.csv', index_col='datetime')

# 2. Separate Features (X) and Targets (y)
X = df[['temperature', 'humidity', 'wind_speed', 'month']] 
y = df[['target_day_1', 'target_day_2', 'target_day_3', 
        'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7']]

# 3. Train/Test Split (80% training, 20% unseen testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Multi-Output Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Performance
predictions = model.predict(X_test)

# Calculate the mean absolute error across the entire 7-day block
mae = mean_absolute_error(y_test, predictions)

print(f"--- Model Metrics ---")
print(f"Average Error across the 7-day forecast: {mae:.2f}°C")

# Optional: Let's see the R2 score for just Day 1 vs Day 7
day_1_r2 = r2_score(y_test['target_day_1'], predictions[:, 0])
day_7_r2 = r2_score(y_test['target_day_7'], predictions[:, 6])
print(f"Day 1 Accuracy (R2): {day_1_r2:.2f}")
print(f"Day 7 Accuracy (R2): {day_7_r2:.2f}")

# 6. Export for the Web App
joblib.dump(model, '7_day_weather_model.pkl')
print("✅ Model successfully saved as '7_day_weather_model.pkl'")