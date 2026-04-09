import streamlit as st
import pandas as pd
import joblib

# 1. Update Page Title and Emoji to reflect New Mansoura
st.set_page_config(page_title="New Mansoura WeatherAI", page_icon="🌊", layout="centered")

# Load the Damietta-trained model (Weather is identical)
model = joblib.load('damietta_7_day_model.pkl')

# 2. Update Header and Description
st.title("🌊 New Mansoura 7-Day Forecaster")
st.markdown("Enter today's atmospheric conditions to generate a predictive 7-day Daily High temperature trend for New Mansoura's coastal climate.")
st.markdown("---")

# 5 columns for inputs
col1, col2, col3, col4, col5 = st.columns(5)

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

input_data = pd.DataFrame({
    'temperature': [temperature],
    'humidity': [humidity],
    'wind_speed': [wind_speed],
    'precipitation': [precipitation],
    'month': [month]
})

st.markdown("<br>", unsafe_allow_html=True)

# 3. Removed st.balloons() for a faster, cleaner feel
if st.button("🚀 Generate 7-Day Forecast", use_container_width=True):
    
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