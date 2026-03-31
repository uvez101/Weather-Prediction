import streamlit as st
import pandas as pd
import joblib

# Updated the theme and title
st.set_page_config(page_title="Damietta WeatherAI", page_icon="🌴", layout="centered")

# Load the NEW Damietta model
model = joblib.load('damietta_7_day_model.pkl')

st.title("New Mansoura 7-Day Forecaster")
st.markdown("Enter today's exact atmospheric conditions to generate a predictive 7-day Daily High temperature trend for the Mediterranean coast.")
st.markdown("---")

# Expand to 5 columns to fit Precipitation
col1, col2, col3, col4, col5 = st.columns(5)

# Notice the default values are now tailored to an Egyptian summer
with col1:
    temperature = st.number_input("Temp (°C)", value=32.0, step=0.5, format="%.1f")
with col2:
    humidity = st.number_input("Humidity (%)", value=65.0, step=1.0, format="%.1f")
with col3:
    wind_speed = st.number_input("Wind (km/h)", value=15.0, step=0.5, format="%.1f")
with col4:
    precipitation = st.number_input("Rain (mm)", value=0.0, step=0.1, format="%.1f")
with col5:
    month = st.number_input("Month", min_value=1, max_value=12, value=8, step=1)

# Ensure the order matches your X features in model.py EXACTLY
input_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'precipitation': [precipitation],
    'month': [month]
})

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Generate 7-Day Forecast", use_container_width=True):
    st.balloons()
    
    predictions = model.predict(input_data)[0] 
    
    st.markdown("### 📈 Predicted Weekly Trend (Daily Highs)")
    
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    chart_data = pd.DataFrame({
        'Daily High (°C)': predictions
    }, index=days)
    
    st.area_chart(chart_data)
    
    st.markdown("### 🗓️ Daily Breakdown")
    metric_cols = st.columns(7)
    for i, col in enumerate(metric_cols):
        col.metric(label=f"Day {i+1}", value=f"{predictions[i]:.1f}°")