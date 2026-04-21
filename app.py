import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np

# 1. Page Config
st.set_page_config(page_title="New Mansoura WeatherAI", page_icon="🌊", layout="centered")

# 2. Rebuild the LSTM Skeleton (Must match Colab EXACTLY)
class WeatherLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=7):
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

# 3. Load the Scaler and the Model
scaler = joblib.load('lstm_scaler.pkl')

model = WeatherLSTM()
# map_location=torch.device('cpu') ensures it runs on Streamlit's free CPU servers, even if trained on a Colab GPU!
model.load_state_dict(torch.load('damietta_lstm_model.pth', map_location=torch.device('cpu')))
model.eval() # Put the model in "evaluation" (prediction) mode

# 4. The UI
st.title("🌊 New Mansoura Deep Learning Forecaster")
st.markdown("Powered by a PyTorch Long Short-Term Memory (LSTM) Neural Network. Enter today's conditions to generate a predictive 7-day trend.")
st.markdown("---")

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

if st.button("🚀 Generate AI Forecast", use_container_width=True):
    
    # 5. Prepare the inputs
    raw_inputs = np.array([[temperature, humidity, wind_speed, precipitation, month]])
    
    # 6. Scale the inputs to match the neural network's training environment
    scaled_inputs = scaler.transform(raw_inputs)
    
    # 7. Convert to a 3D PyTorch Tensor: [Samples=1, Time_Steps=1, Features=5]
    tensor_inputs = torch.FloatTensor(scaled_inputs).unsqueeze(1)
    
    # 8. Predict!
    with torch.no_grad():
        predictions = model(tensor_inputs).numpy()[0]
    
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