import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print("🔍 Booting up the Grid Search Matrix...")

# 1. Load Data
df = pd.read_csv('seattle_forecast_ready.csv', index_col='datetime')
X = df[['temperature', 'humidity', 'wind_speed', 'month']]
y = df[['target_day_1', 'target_day_2', 'target_day_3', 
        'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Base Model
xgb = XGBRegressor(random_state=42)

# 3. Define the Grid (The "Dials" we want to turn)
param_grid = {
    'n_estimators': [50, 100, 200],          # How many trees to build
    'max_depth': [3, 4, 5],                  # How deep/complex the trees can get (lower = less memorization)
    'learning_rate': [0.01, 0.05, 0.1],      # How aggressively it corrects mistakes
    'subsample': [0.7, 0.9, 1.0]             # Percentage of data used per tree (prevents overfitting)
}

# Calculate total combinations: 3 * 3 * 3 * 3 = 81 models!
print("Testing 81 different XGBoost models. You might hear your laptop fans spin up...")

# 4. Setup the Search
# cv=3 means it double-checks its work 3 times per model. n_jobs=-1 uses all your CPU cores.
grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    cv=3, 
    scoring='neg_mean_absolute_error', 
    verbose=1, 
    n_jobs=-1
)

# 5. Let them fight!
grid_search.fit(X_train, y_train)

print("\n🏆 Grid Search Complete!")
print(f"The Winning Combination: {grid_search.best_params_}")

# 6. Test the absolute best model on our unseen test data
champion_model = grid_search.best_estimator_
predictions = champion_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Champion XGBoost MAE: {mae:.2f}°C")