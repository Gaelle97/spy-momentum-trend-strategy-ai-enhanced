"""
data_loader.py
==============
Load SPY daily price data.

Source priority
---------------
  1. Stooq (free, no API key required)
  2. yfinance (requires: pip install yfinance)
  3. Local CSV  data/spy_daily.csv   ← synthetic fallback

IMPORTANT
---------
The synthetic fallback (data/spy_daily.csv) is a regime-calibrated
simulation. Results obtained on synthetic data should NOT be compared
directly to results on real market data. Always prefer real sources (1 or 2).

Usage
-----
    from data_loader import load_spy
    df, is_real = load_spy("2005-01-01", "2024-12-31")
    if not is_real:
        print("WARNING: using synthetic data — results are indicative only.")
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR  = Path(__file__).parent / "data"
CSV_PATH  = DATA_DIR / "spy_daily.csv"


def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["Return"] = df["Close"].pct_change()
    df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def load_spy(start: str = "2005-01-01",
             end:   str = "2024-12-31") -> tuple[pd.DataFrame, bool]:
    """
    Load SPY daily price data.

    Returns
    -------
    df       : DataFrame with columns [Open, High, Low, Close, Volume, Return, LogRet]
    is_real  : True if data is from a real market source (Stooq or yfinance),
               False if synthetic fallback was used.
    """

    # ── 1. Stooq ─────────────────────────────────────────────────────────────
    try:
        d1  = start.replace("-", "")
        d2  = end.replace("-", "")
        url = f"https://stooq.com/q/d/l/?s=spy.us&d1={d1}&d2={d2}&i=d"
        df  = pd.read_csv(url, index_col="Date", parse_dates=True).sort_index()
        df  = _add_returns(df)
        print(f"[data_loader] Stooq OK — {len(df)} rows  (REAL DATA)")
        return df, True
    except Exception as e:
        print(f"[data_loader] Stooq failed: {e}")

    # ── 2. yfinance ──────────────────────────────────────────────────────────
    try:
        import yfinance as yf
        raw = yf.download("SPY", start=start, end=end, progress=False)
        df  = pd.DataFrame()
        df["Close"]  = raw["Adj Close"].squeeze()
        df["Volume"] = raw["Volume"].squeeze()
        df  = _add_returns(df)
        print(f"[data_loader] yfinance OK — {len(df)} rows  (REAL DATA)")
        return df, True
    except Exception as e:
        print(f"[data_loader] yfinance failed: {e}")

    # ── 3. Local CSV synthetic fallback ──────────────────────────────────────
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
        df = _add_returns(df)
        print(f"[data_loader] ⚠  SYNTHETIC DATA — {len(df)} rows")
        print("[data_loader] ⚠  Results are indicative only. Use real data for conclusions.")
        return df, False

    raise FileNotFoundError(
        f"No data source available. Place a real SPY CSV at {CSV_PATH} "
        "or install yfinance (pip install yfinance)."
    )


def generate_synthetic_spy(start: str = "2005-01-01",
                            end:   str = "2024-12-31",
                            seed:  int = 42) -> pd.DataFrame:
    """
    Generate synthetic SPY-like daily prices calibrated to historical regimes.

    Regimes (approximate real SPY periods)
    ----------------------------------------
    2005–2007 : Bull market   mu=+14%/y, vol=10%
    2008–2009 : GFC crash     mu=-35%/y, vol=35%
    2010–2019 : Recovery bull mu=+13%/y, vol=14%
    2020 Q1   : COVID crash   mu=-90%/y, vol=60%
    2020 Q2–  : V-shaped rec  mu=+80%/y, vol=30%
    2021      : Bull          mu=+26%/y, vol=12%
    2022      : Bear          mu=-19%/y, vol=22%
    2023–2024 : Bull          mu=+23%/y, vol=14%

    Returns
    -------
    DataFrame with columns [Close, Volume]
    """
    np.random.seed(seed)
    idx   = pd.bdate_range(start=start, end=end)
    dates = pd.to_datetime(idx)

    regimes = [
        ("2005-01-01", "2007-12-31",  0.14, 0.10),
        ("2008-01-01", "2009-03-31", -0.35, 0.35),
        ("2009-04-01", "2019-12-31",  0.13, 0.14),
        ("2020-01-01", "2020-03-23", -0.90, 0.60),
        ("2020-03-24", "2020-12-31",  0.80, 0.30),
        ("2021-01-01", "2021-12-31",  0.26, 0.12),
        ("2022-01-01", "2022-12-31", -0.19, 0.22),
        ("2023-01-01", "2024-12-31",  0.23, 0.14),
    ]

    price  = 130.0   # SPY approx start price Jan 2005
    prices = []

    for d in dates:
        for r_start, r_end, mu, sigma in regimes:
            if pd.Timestamp(r_start) <= d <= pd.Timestamp(r_end):
                daily_mu    = mu / 252
                daily_sigma = sigma / np.sqrt(252)
                break
        else:
            daily_mu, daily_sigma = 0.08/252, 0.15/np.sqrt(252)

        ret   = np.random.normal(daily_mu, daily_sigma)
        price *= (1 + ret)
        prices.append(max(price, 1.0))

    df = pd.DataFrame({
        "Close":  prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, size=len(dates)),
    }, index=dates)
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    # Generate and save synthetic data for offline use
    DATA_DIR.mkdir(exist_ok=True)
    df_syn = generate_synthetic_spy()
    df_syn.to_csv(CSV_PATH)
    print(f"Synthetic SPY data saved to {CSV_PATH}  ({len(df_syn)} rows)")
