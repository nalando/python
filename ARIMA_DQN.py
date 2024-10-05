import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels import ARIMA

# 取得股票資訊
def get_stock_data(ticker, start_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date)
    return data[['Close', 'Volume']]

def prepare_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data
# 分割數據
data = get_stock_data('TSM', '2000-01-01')
train_data , test_data = prepare_data(data)
print(len(train_data),len(test_data))

def arima_predict(data, order=(1,1,1)):
    model = ARIMA(data, order=order)
    results = model.fit()
    return results.forecast(steps=1)[0]