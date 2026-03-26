"""
strategy.py
===========
Classical Momentum & Trend-Following Strategy on SPY.

Signals
-------
  1. Trend filter  : SMA50 > SMA200
  2. Momentum      : 63-day price return > 0
  3. RSI guard     : RSI(14) < 70

Position sizing
---------------
  Volatility targeting: w_t = clip(sigma* / sigma_hat_t, 0.30, 1.50) * s_t
  sigma* = 12% annualised target
  sigma_hat_t = sqrt(252) * std(r_{t-21:t})   (21-day realised vol)

All positions are lagged one day to prevent look-ahead bias.
Transaction cost: 5 bps per unit of position change.

Usage
-----
    from strategy import build_signals, compute_returns
    df = load_spy(...)           # from data_loader
    df = build_signals(df)
    df = compute_returns(df)
"""

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────
SMA_FAST   = 50
SMA_SLOW   = 200
MOM_WINDOW = 63       # ~3 calendar months
RSI_PERIOD = 14
VOL_WINDOW = 21       # realised vol lookback
VOL_TARGET = 0.12     # 12% annualised
POS_MIN    = 0.30
POS_MAX    = 1.50
COST_BPS   = 0.0005   # 5 bps per unit of position change


# ── Technical indicators ──────────────────────────────────────────────────────

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Price return over `window` trading days."""
    return series.pct_change(window)


def wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's exponential smoothing (alpha = 1/period).
    Standard definition: RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    """
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    alpha  = 1.0 / period

    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def realised_vol(returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Annualised rolling standard deviation of returns."""
    return returns.rolling(window).std() * np.sqrt(252)


# ── Signal construction ───────────────────────────────────────────────────────

def build_signals(df: pd.DataFrame,
                  use_rsi: bool = True,
                  rsi_threshold: float = 70.0) -> pd.DataFrame:
    """
    Add all indicator columns and the composite entry signal to df.

    Parameters
    ----------
    df          : DataFrame with at least a 'Close' and 'Return' column.
    use_rsi     : If False, the RSI filter is disabled (for ablation study).
    rsi_threshold: Overbought threshold (default 70).

    Returns
    -------
    df with added columns: SMA50, SMA200, Momentum, RSI, Signal, RealVol, Weight.
    """
    df = df.copy()

    # ── Indicators ────────────────────────────────────────────────────────────
    df["SMA50"]    = sma(df["Close"], SMA_FAST)
    df["SMA200"]   = sma(df["Close"], SMA_SLOW)
    df["Momentum"] = momentum(df["Close"], MOM_WINDOW)
    df["RSI"]      = wilder_rsi(df["Close"], RSI_PERIOD)
    df["RealVol"]  = realised_vol(df["Return"], VOL_WINDOW)

    # ── Composite binary signal ───────────────────────────────────────────────
    trend_ok    = df["SMA50"] > df["SMA200"]
    momentum_ok = df["Momentum"] > 0

    if use_rsi:
        rsi_ok  = df["RSI"] < rsi_threshold
        signal  = (trend_ok & momentum_ok & rsi_ok).astype(int)
    else:
        signal  = (trend_ok & momentum_ok).astype(int)

    df["Signal"] = signal

    # ── Volatility-targeted position size ─────────────────────────────────────
    raw_weight = (VOL_TARGET / df["RealVol"]).clip(POS_MIN, POS_MAX)
    df["Weight"] = raw_weight * df["Signal"]

    return df.dropna()


# ── Return computation ────────────────────────────────────────────────────────

def compute_returns(df: pd.DataFrame,
                    weight_col: str = "Weight") -> pd.DataFrame:
    """
    Compute net strategy returns using the lagged weight and transaction costs.

    Net return_t = Weight_{t-1} * Return_t - cost_t
    cost_t       = COST_BPS * |Weight_t - Weight_{t-1}|
    """
    df = df.copy()
    w      = df[weight_col]
    w_lag  = w.shift(1)
    cost   = COST_BPS * (w - w_lag).abs()

    df["StratReturn"] = w_lag * df["Return"] - cost
    df["BHReturn"]    = df["Return"]          # Buy & Hold benchmark

    # Cumulative wealth indices (start at 1.0)
    df["StratWealth"] = (1 + df["StratReturn"]).cumprod()
    df["BHWealth"]    = (1 + df["BHReturn"]).cumprod()

    return df.dropna()
