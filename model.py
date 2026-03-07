import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('seattle-weather.csv')

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear

encoder = OneHotEncoder(sparse_output= False, drop='first')
weather_enc = encoder.fit_transform(df[['weather']])

weather_df = pd.DataFrame(weather_enc, columns=encoder.get_feature_names_out(['weather']))

X = pd.concat([df[['precipitation','temp_min', 'wind', 'month']], weather_df], axis=1)
y = df['temp_max']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"--- Model Metrics ---")
print(f"R-Squared (Accuracy Score): {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f} degrees")


#-------------------------deploying and testing-------------------

import joblib

# Save the model and the encoder to files
joblib.dump(model, 'weather_model.pkl')
joblib.dump(encoder, 'weather_encoder.pkl')