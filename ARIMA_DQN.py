!pip install yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import itertools
import warnings
warnings.filterwarnings('ignore')

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
    return result[1] <= 0.05

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
            print('ARIMA%s MSE=%.3f' % (order,mse))
        except:
            continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg

def arima_predict(data, order=(1,1,1)):
    model = ARIMA(data, order=order)
    results = model.fit()
    return results.forecast(steps=1)[0]

# 3. 深度強化學習模型
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 檢查是否有可用的GPU
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Using device: {self.device}")
        
        self.model = self._build_model()

    def _build_model(self):
        with tf.device(self.device):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
                tf.keras.layers.Dense(24, activation='relu'),
                tf.keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        with tf.device(self.device):
            act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        with tf.device(self.device):
            for i in minibatch:
                state, action, reward, next_state, done = self.memory[i]
                target = reward
                if not done:
                    # 將 `next_state` 的形狀調整為 `(1, state_size)`，以便模型能正確處理
                    next_state = np.reshape(next_state, [1, self.state_size])
                    target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

                # 將 `state` 的形狀也調整為 `(1, state_size)`
                state = np.reshape(state, [1, self.state_size])
                target_f = self.model.predict(state)
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
        
        if 1 <= action <= 5:
            buy_percentage = action * 0.1
            buy_amount = int(max_buyable * buy_percentage)
            self.stock_owned += buy_amount
            self.balance -= buy_amount * self.current_price
        elif 6 <= action <= 10:
            sell_percentage = (action - 5) * 0.1
            sell_amount = int(self.stock_owned * sell_percentage)
            self.stock_owned -= sell_amount
            self.balance += sell_amount * self.current_price

        self.current_price = self.price_data[self.index]
        self.current_volume = self.volume_data[self.index]
        new_portfolio_value = self.balance + self.stock_owned * self.current_price
        
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        
        return self._get_state(), reward, False

# 5. 主要訓練和測試循環
def train_and_test(price_data, volume_data, train_ratio=0.8, episodes=100, quick_eval=False):
    if quick_eval:
        data_size = min(1000, len(price_data))
        price_data = price_data[-data_size:]
        volume_data = volume_data[-data_size:]
        episodes = 10

    train_size = int(len(price_data) * train_ratio)
    
    train_prices = price_data[:train_size]
    train_volumes = volume_data[:train_size]
    test_prices = price_data[train_size:]
    test_volumes = volume_data[train_size:]

    env = TradingEnv(train_prices, train_volumes)
    state_size = 4
    action_size = 11
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    episode_rewards = []
    episode_portfolio_values = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # 調整形狀為 `(1, state_size)`
        total_reward = 0
        for time in range(len(train_prices) - 1):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])  # 調整形狀為 `(1, state_size)`
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(env.balance + env.stock_owned * env.current_price)
        print(f"Training episode: {e}/{episodes}, total reward: {total_reward}")

    # 繪製獎勵和投資組合價值的曲線
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(2, 1, 2)
    plt.plot(episode_portfolio_values)
    plt.title('Portfolio Value')
    plt.xlabel('Episode')
    plt.ylabel('Value')

    plt.tight_layout()
    plt.show()

    # 測試環境
    test_env = TradingEnv(test_prices, test_volumes)
    state = test_env.reset()
    state = np.reshape(state, [1, state_size])  # 調整形狀為 `(1, state_size)`
    total_reward = 0
    for time in range(len(test_prices) - 1):
        action = agent.act(state)
        next_state, reward, done = test_env.step(action)
        next_state = np.reshape(next_state, [1, state_size])  # 調整形狀為 `(1, state_size)`
        state = next_state
        total_reward += reward
        if done:
            break
    
    print(f"Test performance - total reward: {total_reward}")
    return agent, total_reward


if __name__ == "__main__":
    tsmc_data = get_stock_data("TSM", "2000-01-01")
    train_data, test_data = prepare_data(tsmc_data, train_ratio=0.8)

    print("Performing quick evaluation...")
    quick_agent, quick_reward = train_and_test(tsmc_data['Close'].values, tsmc_data['Volume'].values, quick_eval=True)
    print(f"Quick evaluation reward: {quick_reward}")

    print("Performing full training...")
    trained_agent, test_reward = train_and_test(tsmc_data['Close'].values, tsmc_data['Volume'].values)
    print(f"Full training test reward: {test_reward}")