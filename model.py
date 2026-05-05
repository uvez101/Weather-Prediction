import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

print("🚀 Booting up Omni-Model GPU Environment...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the massive new dataset
df = pd.read_csv('damietta_forecast_ready.csv', index_col='datetime')

# 2. X now has 6 inputs (we added temp_min!)
X = df[['temp_max', 'temp_min', 'humidity', 'wind_speed', 'precipitation', 'month']].values

# Dynamically grab all 28 target columns (anything with 'day_' in the name)
target_cols = [col for col in df.columns if 'day_' in col]
y = df[target_cols].values

# 3. Scale the Data (The scaler now learns how to scale all 6 inputs)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for LSTM: [Samples, Time_Steps, Features]
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 4. The Upgraded Deep Learning Architecture
class WeatherLSTM(nn.Module):
    # Notice the new input_size and output_size!
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, output_size=28):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # The Decision Layer now outputs 28 numbers
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# We slightly increased the 'hidden_size' brain capacity (128) because the task is 4x harder now
model = WeatherLSTM(input_size=6, hidden_size=128, num_layers=2, output_size=28).to(device)

# 5. Define Loss and Optimizer
criterion = nn.L1Loss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. The Training Loop
epochs = 1500
print(f"\n🔥 Training the 28-Target Omni-Model on {device}...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Overall Loss (MAE): {loss.item():.4f}')

# 7. Evaluate
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"\n🏆 Final Test MAE (across all 28 variables): {test_loss.item():.2f}")

# 8. Save the new 6-input Scaler and 28-output Model
torch.save(model.state_dict(), 'damietta_lstm_omni.pth')
joblib.dump(scaler_X, 'lstm_scaler_omni.pkl')
print("\n✅ Saved 'damietta_lstm_omni.pth' and 'lstm_scaler_omni.pkl'")