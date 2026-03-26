import pandas as pd
import numpy as np
from strategy import compute_signals

def ewma_volatility(returns, lambda_=0.94):
    """
    Calcul de la volatilité exponentiellement pondérée (EWMA).
    """
    vol2 = np.zeros(len(returns))
    vol2[0] = returns.iloc[0]**2
    for t in range(1, len(returns)):
        vol2[t] = lambda_ * vol2[t-1] + (1-lambda_) * returns.iloc[t]**2
    return np.sqrt(vol2)

def backtest(prices, target_vol=0.12):
    """
    Backtest de la stratégie momentum + trend-following avec
    volatility targeting.
    """
    signals = compute_signals(prices)
    daily_returns = prices.pct_change().fillna(0)
    vol = ewma_volatility(daily_returns)
    position_size = np.clip(target_vol / vol, 0.3, 1.5)
    
    strategy_returns = daily_returns * signals.shift(1) * position_size
    equity_curve = (1 + strategy_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    return equity_curve, drawdown
