import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

print("🚀 Booting up PyTorch GPU Environment...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 1. Load Data
df = pd.read_csv('damietta_forecast_ready.csv', index_col='datetime')
X = df[['temperature', 'humidity', 'wind_speed', 'precipitation', 'month']].values
y = df[['target_day_1', 'target_day_2', 'target_day_3', 
        'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7']].values

# 2. Scale the Data (CRITICAL FOR NEURAL NETWORKS)
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Reshape for LSTM: [Samples, Time_Steps, Features]
# We are currently using 1 time step (today's weather) to predict the future.
X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 4. Build the Deep Learning Architecture
class WeatherLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=7):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The Memory Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # The Decision Layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = WeatherLSTM().to(device)

# 5. Define Loss (MAE) and Optimizer (Adam)
criterion = nn.L1Loss() # L1Loss is the PyTorch equivalent of Mean Absolute Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. The Training Loop (Epochs)
epochs = 1500
print("\n🔥 Starting Deep Learning Training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss (MAE): {loss.item():.4f}°C')

# 7. Evaluate on Test Data
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    print(f"\n🏆 Final Test MAE: {test_loss.item():.2f}°C")

# 8. Save the Model and the Scaler
torch.save(model.state_dict(), 'damietta_lstm_model.pth')
joblib.dump(scaler_X, 'lstm_scaler.pkl')
print("\n✅ Saved 'damietta_lstm_model.pth' and 'lstm_scaler.pkl'")