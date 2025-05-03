import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Fetch stock data
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
stock = yf.Ticker(ticker)
data = stock.history(start=start_date, end=end_date)

# Scale close price
data_close = data['Close'].copy()
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(data_close).reshape(-1, 1))

# Dataset creation function
def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        dataX.append(dataset[i:(i + time_step), 0])
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Create train/test datasets
time_step = 100
training_size = int(len(df1) * 0.8)
train_data = df1[:training_size]
test_data = df1[training_size:]
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train).float().unsqueeze(-1)  # [samples, time_step, 1]
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float().unsqueeze(-1)
y_test = torch.tensor(y_test).float()

# Dataset and DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=64, shuffle=False)

# LSTM + Attention model
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, num_layers=2):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn_fc = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def attention(self, lstm_output):
        # lstm_output: [batch, seq_len, hidden_dim]
        attn_weights = torch.softmax(self.attn_fc(lstm_output), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_output, dim=1)  # [batch, hidden_dim]
        return context

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        context = self.attention(lstm_out)  # [batch, hidden_dim]
        out = self.fc(context)  # [batch, 1]
        return out.squeeze()  # [batch]

# Initialize model, loss, and optimizer
model = LSTMWithAttention()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.6f}")
