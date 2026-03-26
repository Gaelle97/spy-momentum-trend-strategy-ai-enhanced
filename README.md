# spy-momentum-trend-strategy-ai-enhanced
Systematic momentum &amp; trend-following strategy on SPY (2005–2024). Combines moving averages, momentum, and RSI filters with volatility targeting to reduce drawdowns (–17% vs –42%) and improve risk-adjusted returns. Includes backtesting, Monte Carlo simulation, and options pricing.
Momentum & Trend-Following Strategy on SPY

This project implements a systematic quantitative investment strategy on the S&P 500 ETF (SPY), designed to reduce drawdowns while maintaining solid long-term performance.

Strategy Overview
The model combines three well-documented market signals:
Trend filter: 50-day moving average > 200-day moving average
Momentum: Price higher than 3 months ago
Overbought filter: RSI below 70

Investment is only made when all three conditions are met.

Risk Management
Instead of staying fully invested, the strategy applies volatility targeting:
Target volatility: 12% annually
Position sizing dynamically adjusted:
Position = clip( Target Vol / Current Vol, 30%, 150% )
This is the key driver behind drawdown reduction.

📊 Results (2005–2024)
Metric	Strategy	Buy & Hold
Annual Return	9.1%	17.0%
Volatility	9.4%	19.6%
Sharpe Ratio	0.54	0.66
Max Drawdown	-17%	-42%
Calmar Ratio	0.53	0.41

✅ -60% drawdown reduction
✅ Much smoother equity curve
⚠️ Lower absolute return (by design)

Methodology
Data: SPY daily prices (2005–2024)
Synthetic regime-calibrated dataset
Volatility estimation: EWMA (RiskMetrics)
Transaction costs: 5 bps per trade

Additional Features
Monte Carlo simulations (1,000 scenarios)
Options pricing via Black-Scholes
Full backtest pipeline in Python

Limitations
Potential overfitting (50/200 MA widely known)
Regime dependency (underperforms in sideways markets)
Tail risk underestimation
SPY-specific bias (US market dominance)

Tech Stack
Python
pandas / numpy
scipy
matplotlib

Paper
Full research paper available in the repository.

Disclaimer
This project is for research and educational purposes only.
It does not constitute financial advice.
