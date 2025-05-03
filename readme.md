# 📈 Stock Price Prediction using LSTM with Attention

This project predicts the stock closing price of **Apple Inc. (AAPL)** using a deep learning model based on **LSTM (Long Short-Term Memory)** with an **attention mechanism**, implemented in **PyTorch**.

---

##  Overview

This model leverages time-series data and deep learning to forecast stock prices. The pipeline includes:

- Fetching historical stock data from Yahoo Finance
- Scaling data using `MinMaxScaler`
- Creating sequences of time steps
- Building a custom LSTM model with attention
- Training the model using Mean Squared Error (MSE)
- Evaluating the model using Root Mean Squared Error (RMSE)

---

##  Model Architecture

- **Input Layer**: Stock closing prices with time steps
- **LSTM Layers**: Two stacked LSTM layers with 50 hidden units
- **Attention Layer**: Assigns weights to different time steps
- **Fully Connected Layer**: Outputs the predicted stock price

---

##  Result

After 200 epochs of training, the model achieved the following performance on test data:

**RMSE (Root Mean Squared Error): 6.6702**

---

##  Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
