import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

# 1. Page Config
st.set_page_config(page_title="New Mansoura Omni-Forecaster", page_icon="🌊", layout="wide")

# 2. Rebuild the Upgraded Omni-LSTM Skeleton
class WeatherLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=28):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 3. Load the Omni-Scaler and the Omni-Model
scaler = joblib.load('lstm_scaler_omni.pkl')

model = WeatherLSTM()
model.load_state_dict(torch.load('damietta_lstm_omni.pth', map_location=torch.device('cpu')))
model.eval()

# 4. The UI Header
st.title("🌊 New Mansoura Omni-Forecaster")
st.markdown("Powered by a Multi-Task PyTorch LSTM. Enter today's exact conditions to predict the full atmospheric profile for the next 7 days.")
st.markdown("---")

# 5. The Input Matrix (Now featuring 6 columns)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    temp_max = st.number_input("Max Temp (°C)", value=32.0, step=0.5)
with col2:
    temp_min = st.number_input("Min Temp (°C)", value=22.0, step=0.5)
with col3:
    humidity = st.number_input("Humidity (%)", value=65.0, step=1.0)
with col4:
    wind_speed = st.number_input("Wind (km/h)", value=15.0, step=0.5)
with col5:
    precipitation = st.number_input("Rain (mm)", value=0.0, step=0.1)
with col6:
    month = st.number_input("Month", min_value=1, max_value=12, value=8, step=1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Generate Omni-Forecast", use_container_width=True):
    
    # 6. Prepare inputs EXACTLY as the scaler expects them
    raw_inputs = np.array([[temp_max, temp_min, humidity, wind_speed, precipitation, month]])
    scaled_inputs = scaler.transform(raw_inputs)
    tensor_inputs = torch.FloatTensor(scaled_inputs).unsqueeze(1)
    
    # 7. Predict all 28 variables!
    with torch.no_grad():
        raw_predictions = model(tensor_inputs).numpy()[0]
        
    # 8. Reshape the 28 numbers into a 7x4 Grid
    # The network outputs [max_day1, min_day1, hum_day1, wind_day1, max_day2...]
    grid_predictions = raw_predictions.reshape(7, 4)
    
    days = [f'Day {i}' for i in range(1, 8)]
    df_preds = pd.DataFrame(grid_predictions, columns=['Max Temp (°C)', 'Min Temp (°C)', 'Humidity (%)', 'Wind (km/h)'], index=days)
    
    st.success("✅ Multi-Task Physics Forecast Generated Successfully!")
    
    # 9. Build a Pro-Level Tabbed Dashboard
    tab1, tab2, tab3 = st.tabs(["🌡️ Temperatures", "💧 Humidity", "💨 Wind Speed"])
    
    with tab1:
        st.subheader("7-Day Temperature Range")
        st.line_chart(df_preds[['Max Temp (°C)', 'Min Temp (°C)']])
        
    with tab2:
        st.subheader("7-Day Humidity Trend")
        st.area_chart(df_preds[['Humidity (%)']], color="#4A90E2")
        
    with tab3:
        st.subheader("7-Day Wind Gusts")
        st.bar_chart(df_preds[['Wind (km/h)']], color="#50E3C2")
        
    # 10. The Raw Data Matrix
    st.markdown("### 🗓️ Daily Data Matrix")
    st.dataframe(df_preds.style.format("{:.1f}"), use_container_width=True)