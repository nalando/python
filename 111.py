import numpy as np
import pandas as pd
import itertools
import yfinance as yf
from keras import layers, models, optimizers
from keras.models import Sequential
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import random

# 1. 數據獲取和準備
def get_stock_data(ticker, start_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date)
    return data[['Close', 'Volume']]

def prepare_data(data, train_ratio=0.8):
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# 2. ARIMA模型和測試
def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    return result[1] <= 0.05  # p-value <= 0.05 表示數據是平穩的

def difference(dataset):
    return dataset.diff().dropna()

def evaluate_arima_model(train, test, arima_order):
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(train, test, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p,d,q)
        try:
            mse = evaluate_arima_model(train, test, order)
            if mse < best_score:
                best_score, best_cfg = mse, order
            print(f'ARIMA{order} MSE={mse:.3f}')
        except:
            continue
    print(f'Best ARIMA{best_cfg} MSE={best_score:.3f}')
    return best_cfg

def arima_predict(data, order=(1,1,1)):
    model = ARIMA(data, order=order)
    results = model.fit()
    return results.forecast(steps=1).iloc[0]

# 3. 深度強化學習模型
class DQNAgent:
    def __init__(self, state_size, action_size, max_memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.max_memory_size = max_memory_size
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 4. 交易環境
class TradingEnv:
    def __init__(self, price_data, volume_data, initial_balance=100000, max_stock_quantity=1000):
        self.price_data = price_data
        self.volume_data = volume_data
        self.initial_balance = initial_balance
        self.max_stock_quantity = max_stock_quantity
        self.reset()

    def reset(self):
        self.index = 0
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.current_price = self.price_data[self.index]
        self.current_volume = self.volume_data[self.index]
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.balance / self.initial_balance,
            self.stock_owned / self.max_stock_quantity,
            self.current_price / np.max(self.price_data),
            self.current_volume / np.max(self.volume_data)
        ])

    def step(self, action):
        self.index += 1
        if self.index >= len(self.price_data) - 1:
            return self._get_state(), 0, True

        old_portfolio_value = self.balance + self.stock_owned * self.current_price
        max_buyable = min(self.current_volume, self.balance // self.current_price, self.max_stock_quantity - self.stock_owned)
        
        if action == 1:
            buy_amount = int(max_buyable * 0.1)
            self.stock_owned += buy_amount
            self.balance -= buy_amount * self.current_price
        elif action == 2:
            buy_amount = int(max_buyable * 0.2)
            self.stock_owned += buy_amount
            self.balance -= buy_amount * self.current_price
        elif action == 3:
            sell_amount = int(self.stock_owned * 0.1)
            self.stock_owned -= sell_amount
            self.balance += sell_amount * self.current_price
        elif action == 4:
            sell_amount = int(self.stock_owned * 0.2)
            self.stock_owned -= sell_amount
            self.balance += sell_amount * self.current_price

        self.current_price = self.price_data[self.index]
        self.current_volume = self.volume_data[self.index]
        new_portfolio_value = self.balance + self.stock_owned * self.current_price
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value

        return self._get_state(), reward, False

# 5. 主要訓練和測試循環
def train_and_test(price_data, volume_data, train_ratio=0.8, episodes=100):
    train_size = int(len(price_data) * train_ratio)
    train_prices = price_data[:train_size]
    train_volumes = volume_data[:train_size]
    test_prices = price_data[train_size:]
    test_volumes = volume_data[train_size:]

    env = TradingEnv(train_prices, train_volumes)
    state_size = 4
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    # 訓練階段
    for e in range(episodes):
        state = np.reshape(env.reset(), [1, state_size])
        total_reward = 0
        for time in range(len(train_prices) - 1):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = np.reshape(next_state, [1, state_size])
            if done:
                break
            agent.replay(batch_size)
        print(f'Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}')

    # 測試階段
    test_env = TradingEnv(test_prices, test_volumes)
    state = np.reshape(test_env.reset(), [1, state_size])
    total_reward = 0
    for time in range(len(test_prices) - 1):
        action = agent.act(state)
        next_state, reward, done = test_env.step(action)
        total_reward += reward
        state = np.reshape(next_state, [1, state_size])
        if done:
            break
    print(f'Test Reward: {total_reward:.2f}')
