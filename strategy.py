import pandas as pd
import numpy as np

def compute_signals(prices):
    """
    Compute entry signals: trend, momentum, RSI.
    """
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    momentum = prices / prices.shift(63)  # 3 months ~ 63 trading days
    rsi = compute_rsi(prices)
    
    signal = (ma50 > ma200) & (momentum > 1) & (rsi < 70)
    return signal

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
