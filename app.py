import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved assets
model = joblib.load('weather_model.pkl')
encoder = joblib.load('weather_encoder.pkl')

st.title("☀️ Temperature Predictor")
st.write("Adjust the weather conditions to see the predicted Max Temperature.")

# 1. Setup User Inputs in a Sidebar
with st.sidebar:
    st.header("Input Conditions")
    temp_min = st.slider("Min Temperature (°C)", -5.0, 30.0, 10.0)
    precipitation = st.number_input("Precipitation (mm)", 0.0, 100.0, 0.0)
    wind = st.slider("Wind Speed", 0.0, 20.0, 5.0)
    month = st.selectbox("Month", list(range(1, 13)))
    weather_type = st.selectbox("Weather Type", ['drizzle', 'rain', 'sun', 'snow', 'fog'])

# 2. Preprocess the User Input
# Create a small dataframe for the categorical encoding
weather_input = pd.DataFrame({'weather': [weather_type]})
weather_encoded = encoder.transform(weather_input)
weather_encoded_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather']))

# Combine all inputs into one row
input_data = pd.DataFrame({
    'precipitation': [precipitation],
    'temp_min': [temp_min],
    'wind': [wind],
    'month': [month]
})
final_X = pd.concat([input_data, weather_encoded_df], axis=1)

# 3. Predict!
if st.button("Predict Max Temp"):
    prediction = model.predict(final_X)
    st.metric("Predicted Max Temperature", f"{prediction[0]:.2f} °C")