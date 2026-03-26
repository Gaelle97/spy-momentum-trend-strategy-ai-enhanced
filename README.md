spy-momentum-trend-strategy-ai-enhanced

Systematic momentum & trend-following strategy on SPY (2005–2024) with optional ML enhancement.

Combines moving averages, momentum, and RSI filters with volatility targeting.
Honest assessment of limitations included. Walk-forward validation, RSI ablation study, and multi-benchmark comparison.

---

## What This Strategy Is (and Is Not)

> TL;DR: This is an equity timing + volatility scaling strategy.
>
> Its value is **not** beating the market on raw return.
> Its value is **improving the investor journey** — dramatically reducing drawdowns,
> halving portfolio volatility, and making the portfolio more holdable through crises.

| Investor type | Fit |
|---|---|
| Family office / risk-conscious wealth management | ✅ Strong fit |
| Disciplined retail investor (long horizon) | ✅ Good fit |
| Risk overlay on a broader portfolio | ✅ Good fit |
| Benchmark-relative institutional mandate | ⚠️  Not suitable |
| Pure alpha seeker (wants to beat buy-and-hold) | ❌ Wrong strategy |

---

Results Summary (2005–2024)

| Metric | Classical | ML-Enhanced | Buy & Hold | 60/40 | Vol-Target Only |
|--------|-----------|-------------|------------|-------|-----------------|
| Annual Return | 9.1% | 10.2% | 17.0% | 8.5% | 8.8% |
| Ann. Volatility | 9.4% | 8.5% | 19.6% | 11.2% | 9.1% |
| **Sharpe Ratio** | **0.54** | **0.62** | **0.66** | 0.58 | 0.56 |
| Max Drawdown | **–17%** | **–14.5%** | –42% | –28% | –19% |
| Calmar Ratio | **0.53** | **0.70** | 0.41 | 0.30 | 0.46 |

Honest reading:
- ✅ Drawdown reduced ~60% vs buy-and-hold — the primary objective
- ✅ Calmar ratio 70% higher than buy-and-hold (ML version)
- ✅ Volatility halved — far smoother portfolio to hold
- ⚠️  Sharpe 0.54 < buy-and-hold 0.66 — not more *efficient* on raw risk/return
- ⚠️  Lower absolute return — the strategy is not always invested (by design)
- ⚠️  Results on **synthetic data** — see Data section

> **Why does Vol-Targeting Only score comparably?**
> This is the benchmark that isolates whether the trend/momentum SIGNALS add value
> beyond simply scaling by volatility. If it matches the classical strategy, the
> signals contribute little. This is intentional and honest.

---

Strategy Architecture

1. Classical Rules (must ALL be true to invest)

```
① SMA(50) > SMA(200)    — confirmed long-term uptrend
② Return(3M) > 0        — recent momentum is positive
③ RSI(14) < 70          — market is not in overbought territory
```

2. Volatility Targeting

```
Position = clip( 12% target vol / current realised vol,  30%,  150% )
```

This is **the single most important feature** of the strategy.
During high-volatility regimes (2008, March 2020), exposure is mechanically reduced
without any forward-looking information. This explains the –17% max drawdown vs –42%.

3. Transaction Costs

```
Cost per trade = 5 bps × |change in position|
```

Conservative for today's SPY spreads. Likely optimistic for 2005 conditions.
No slippage modelled — a known limitation.

4. ML Enhancement (Random Forest)

A Random Forest classifier predicts whether SPY will be higher in 63 trading days.
A position is entered only when **both** classical rules AND the ML model agree.

Honest caveats on the ML layer:
- Walk-forward OOS accuracy: typically **52–56%** — marginally above random
- Gains (+0.08 Sharpe) obtained on synthetic data; expect smaller improvement on real data
- The double-filter mostly reduces trade frequency, not necessarily adds predictive edge
- Risk of overfitting without strict temporal cross-validation

---

Limitations

| Limitation | Severity | Mitigation |
|---|---|---|
| Synthetic data used | 🔴 High | Rerun on real Stooq/yfinance data |
| Signals are widely known (MA50/200) | 🟡 Medium | Walk-forward validation included |
| RSI may add noise, not edge | 🟡 Medium | Ablation study included |
| 5 bps cost optimistic for 2005 | 🟡 Medium | Sensitivity to cost assumptions |
| No slippage modelled | 🟡 Medium | Pro-grade: add market impact model |
| SPY-specific (US equity bias) | 🟡 Medium | Test on EFA, GLD, TLT |
| Fat tails underestimated | 🟡 Medium | Use historical simulation VaR |
| ML OOS accuracy ~54% | 🟡 Medium | Feature importance analysis included |

---

Walk-Forward Validation

Training period: 2005–2014 | Test period: 2015–2024

This is the minimum bar for a credible systematic strategy.
Results are generated automatically in `backtest.py`.

**Interpretation:** If OOS Sharpe is close to IS Sharpe → strategy is robust.
If OOS is much lower → the rules are overfit to the training period.

---

RSI Ablation Study

The RSI filter is tested with and without:
```
python backtest.py   # prints RSI ablation table
```
If "Without RSI" ≥ "With RSI" on Sharpe/Calmar → the filter adds noise, simplify the model.

---

Data

- **Primary**: Stooq.com (free, no API key) — real adjusted daily prices
- **Fallback**: yfinance (`pip install yfinance`)
- **Last resort**: `data/spy_daily.csv` — synthetic regime-calibrated data

> ⚠️  **All paper results use synthetic data. This limits credibility.**
> Run `python data_loader.py` to generate synthetic data,
> or use Stooq/yfinance for real data (recommended).

---

Tech Stack

| Library | Role |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Time series data management |
| `scipy` | Statistical analysis |
| `matplotlib` | All visualisations |
| `scikit-learn` | Random Forest classifier |
| `yfinance` *(optional)* | Real SPY data download |

```bash
pip install -r requirements.txt
```

---

Usage

```bash
# 1. Generate synthetic data (if no real data available)
python data_loader.py

# 2. Run full backtest (classical strategy + walk-forward + RSI ablation)
python backtest.py

# 3. Generate all figures
python generate_analysis.py

# 4. Run ML-enhanced strategy
python ml_strategy.py
```

---

Project Structure

```
spy-momentum-trend-strategy-ai-enhanced/
├── README.md                # This file
├── LICENSE                  # MIT License
├── .gitignore
├── requirements.txt
│
├── data_loader.py           # SPY data loading (Stooq → yfinance → synthetic)
├── strategy.py              # Classical strategy: signals + vol-targeting + returns
├── backtest.py              # Full backtest: benchmarks, walk-forward, RSI ablation
├── ml_strategy.py           # Walk-forward Random Forest ML enhancement
├── generate_analysis.py     # All figures: wealth, drawdown, MC, walk-forward, ablation
│
├── notebooks/
│   └── analysis.ipynb       # Interactive exploration
│
├── data/
│   └── spy_daily.csv        # Synthetic SPY data (real data not committed)
│
├── figures/                 # Generated charts (auto-populated)
│   ├── 01_wealth_curves.png
│   ├── 02_drawdown_profile.png
│   ├── 03_rolling_sharpe.png
│   ├── 04_return_distribution.png
│   ├── 05_monte_carlo.png
│   ├── 06_walk_forward.png
│   ├── 07_rsi_ablation.png
│   └── dashboard.png
│
└── paper.pdf                # Research paper
```

---

Next Steps (Roadmap)

- [ ] Rerun on **real SPY data** (Stooq) — critical for credibility
- [ ] **Multi-asset test** (EFA, GLD, TLT, AGG) — if it works cross-asset, much stronger
- [ ] **Stricter cost model** (10–15 bps, slippage simulation)
- [ ] **ML feature importance** analysis — does the RF rediscover momentum?
- [ ] **HMM regime detector** — improve signal in sideways markets
- [ ] **Parameter sensitivity** — test SMA windows 30/100, 60/200 etc.

---

References

1. Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
2. Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Black, F. & Scholes, M. (1973). Pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
5. J.P. Morgan / Reuters (1996). *RiskMetrics Technical Document*, 4th ed.

---

Disclaimer

This project is for **research and educational purposes only**.
It does not constitute financial advice.
Past performance (including simulated performance) is not indicative of future results.
Synthetic data results are illustrative only and should not be used for investment decisions.
