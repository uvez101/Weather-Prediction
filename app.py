import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="WeatherAI Forecast", page_icon="🌤️", layout="centered")

# Load the NEW Random Forest model
model = joblib.load('7_day_weather_model.pkl')

st.title("🌤️ Next-Gen Weather Forecaster")
st.markdown("Enter today's exact atmospheric conditions and the current month to generate a predictive 7-day temperature trend.")
st.markdown("---")

# 1. Expand to 4 columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    temperature = st.number_input("Temp (°C)", value=15.0, step=0.5, format="%.1f")
with col2:
    humidity = st.number_input("Humidity (%)", value=75.0, step=1.0, format="%.1f")
with col3:
    wind_speed = st.number_input("Wind (m/s)", value=3.0, step=0.5, format="%.1f")
with col4:
    # 2. Add the Month input (Restricted between 1 and 12)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=3, step=1)

# 3. Add 'month' to the DataFrame so it matches the model's training data exactly
input_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'month': [month]
})

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Generate 7-Day Forecast", use_container_width=True):
    
    predictions = model.predict(input_data)[0] 
    
    st.markdown("### 📈 Predicted Weekly Trend")
    
    days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    chart_data = pd.DataFrame({
        'Temperature (°C)': predictions
    }, index=days)
    
    st.area_chart(chart_data)
    
    st.markdown("### 🗓️ Daily Breakdown")
    metric_cols = st.columns(7)
    for i, col in enumerate(metric_cols):
        col.metric(label=f"Day {i+1}", value=f"{predictions[i]:.1f}°")