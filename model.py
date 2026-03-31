import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print("🧠 Booting up the New Damietta training sequence...")

# 1. Load the new Egyptian dataset
df = pd.read_csv('damietta_forecast_ready.csv', index_col='datetime')

# 2. ADD PRECIPITATION TO INPUTS!
X = df[['temperature', 'humidity', 'wind_speed', 'precipitation', 'month']]
y = df[['target_day_1', 'target_day_2', 'target_day_3', 
        'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Use the Champion Hyperparameters from our Grid Search
model = XGBRegressor(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=3, 
    subsample=0.7, 
    random_state=42
) 
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"--- Model Metrics ---")
print(f"Average Error across the 7-day forecast: {mae:.2f}°C")

# 5. Export with a new name so we don't confuse it with Seattle
joblib.dump(model, 'damietta_7_day_model.pkl')
print("✅ Model successfully saved as 'damietta_7_day_model.pkl'")