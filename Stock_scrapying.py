import requests
from bs4 import BeautifulSoup
import time

def get_stock_price(symbol):
    # Yahoo Finance 股票頁面的 URL 模板
    url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}&.tsrc=fin-srch"
    
    # 發送HTTP请求
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    # 確保請求成功
    if response.status_code != 200:
        print(f"Failed to retrieve data for {symbol}")
        return None
    
    # 解析HTML內容
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 尋找包含股價的標籤
    stock_price_element = soup.find('fin-streamer', {'data-symbol': symbol, 'data-field': 'regularMarketPrice'})
    
    # 確認找到股價元素
    if stock_price_element:
        stock_price = stock_price_element.text
        return stock_price
    else:
        print(f"Stock price element not found for {symbol}")
        return None


def get_multiple_stocks(symbols):
    prices = {}
    for symbol in symbols:
        price = get_stock_price(symbol)
        if price:
            prices[symbol] = price
    return prices


# 目標股票的代號
stock_symbols = ['AAPL', 'TSM', 'TSLA', 'NVDA']

# 設置抓取的間隔時間 (秒)
interval = 10

# 設置抓取的次數 (例如: 5次)
iterations = 5

for i in range(iterations):
    print(f"Iteration {i+1}:")
    
    # 抓取所有股票的股價
    stock_prices = get_multiple_stocks(stock_symbols)
    
    # 顯示股價
    for symbol, price in stock_prices.items():
        print(f"{symbol}: {price}")
    
    print("-" * 40)
    
    # 每次抓取後等待 interval 秒
    time.sleep(interval)

print("Finished grabbing stock prices.")
