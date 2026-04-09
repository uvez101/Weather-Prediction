import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print("🔍 Booting up the Damietta Grid Search Matrix...")

# 1. Load the NEW Damietta Data
df = pd.read_csv('damietta_forecast_ready.csv', index_col='datetime')

# 2. Add 'precipitation' to the feature list!
X = df[['temperature', 'humidity', 'wind_speed', 'precipitation', 'month']]
y = df[['target_day_1', 'target_day_2', 'target_day_3', 
        'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define the Base Model
xgb = XGBRegressor(random_state=42)

# 4. Define the Grid
param_grid = {
    'n_estimators': [50, 100, 200],          
    'max_depth': [3, 4, 5],                  
    'learning_rate': [0.01, 0.05, 0.1],      
    'subsample': [0.7, 0.9, 1.0]             
}

print("Testing 81 different XGBoost models on Egyptian weather patterns...")

# 5. Setup the Search
grid_search = GridSearchCV(
    estimator=xgb, 
    param_grid=param_grid, 
    cv=3, 
    scoring='neg_mean_absolute_error', 
    verbose=1, 
    n_jobs=-1
)

# 6. Let them fight!
grid_search.fit(X_train, y_train)

print("\n🏆 Grid Search Complete!")
print(f"The Winning Combination: {grid_search.best_params_}")

# 7. Test the absolute best model
champion_model = grid_search.best_estimator_
predictions = champion_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Champion XGBoost MAE: {mae:.2f}°C")