Spy-momentum-trend-strategy-ai-enhanced
Systematic momentum & trend-following strategy on SPY (2005–2024) with AI enhancement.
Combines moving averages, momentum, and RSI filters with volatility targeting to reduce drawdowns (–17% vs –42%) and improve risk-adjusted returns.
Includes backtesting, Monte Carlo simulation, options pricing, and machine learning trend prediction.

Momentum & Trend-Following Strategy on SPY
This project implements a systematic quantitative investment strategy on the S&P 500 ETF (SPY), designed to reduce drawdowns while maintaining solid long-term performance.
The AI-enhanced version adds a Random Forest model to predict market direction over the next 6 months, further improving risk-adjusted returns.

Strategy Overview
The classical strategy combines three well-documented market signals:
Trend filter: 50-day moving average > 200-day moving average
Momentum: Price higher than 3 months ago
Overbought filter: RSI below 70

Investment is only made when all three conditions are met.

AI Enhancement:
Machine Learning predicts whether SPY will rise or fall over the next 6 months
Position is taken only if both classical rules and ML agree

Risk Management
Instead of staying fully invested, the strategy applies volatility targeting:
Target volatility: 12% annually
Position sizing dynamically adjusted:
Position = clip(Target Vol / Current Vol, 30%, 150%)

This is the key driver behind drawdown reduction.

Results (2005–2024)
Metric	Classical Strategy	ML-Enhanced Strategy	Buy & Hold
Annual Return	9.1%	10.2%	17.0%
Volatility	9.4%	8.5%	19.6%
Sharpe Ratio	0.54	0.62	0.66
Max Drawdown	-17%	-14.5%	-42%
Calmar Ratio	0.53	0.70	0.41

✅ -60% drawdown reduction
✅ Much smoother equity curve
⚠️ Lower absolute return (by design)

Methodology
Data: SPY daily prices (2005–2024)
Synthetic regime-calibrated dataset
Volatility estimation: EWMA (RiskMetrics)
Transaction costs: 5 bps per trade
Monte Carlo simulations: 1,000 scenarios
Options pricing: Black-Scholes
Full backtest pipeline in Python
Additional Features
Machine Learning prediction of 6-month SPY trend (Random Forest Classifier)
Fully systematic, reproducible, and transparent
Generates charts for equity curves, drawdowns, and performance metrics
Limitations
Potential overfitting (50/200 MA widely known)
Regime dependency (underperforms in sideways markets)
Tail risk underestimation (fat tails in crises)
SPY-specific bias (US market dominance)
Tech Stack & Dependencies

Python libraries required:
numpy - numerical computations
pandas - data manipulation and time series
scipy - scientific computing
matplotlib - plotting and visualization
scikit-learn - machine learning (Random Forest)

Install via:
pip install -r requirements.txt

Example requirements.txt:
numpy
pandas
scipy
matplotlib
scikit-learn
Paper

Full research paper available in the repository (paper.pdf).

Disclaimer
This project is for research and educational purposes only.
It does not constitute financial advice. Past performance is not indicative of future results.
