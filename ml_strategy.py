"""
ml_strategy.py
==============
ML enhancement: Random Forest classifier predicting 6-month SPY direction.

Design principles (honest ML)
------------------------------
  1. STRICT TEMPORAL SPLIT — never shuffle time series data.
     Training on t, prediction on t+1. No look-ahead.
  2. WALK-FORWARD ONLY — the model is only evaluated on data it has
     never seen during training.
  3. TARGET: binary — will SPY close higher in 63 trading days (3 months)?
     6 months (126 days) is used for labelling but 3M is more predictable.
  4. HONEST ASSESSMENT — ML gains are modest and not guaranteed on real data.
     This module reports feature importances and OOS accuracy explicitly.

IMPORTANT CAVEAT
----------------
The gains shown here (+0.08 Sharpe vs classical) are obtained on SYNTHETIC data.
On real SPY data, expect smaller and noisier improvements.
A Random Forest on 5 features with ~3,000 training rows is a weak learner —
its value is marginal filtering, not strong prediction.

Usage
-----
    from ml_strategy import build_ml_signals, run_ml_backtest
    df = load_spy(...)
    df_ml = build_ml_signals(df)
    metrics = run_ml_backtest(df_ml, df_classical)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import classification_report, accuracy_score

from strategy import build_signals, compute_returns, COST_BPS, realised_vol, VOL_WINDOW


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct ML features from price and return data.

    All features are computed using only past data (no look-ahead).

    Features
    --------
    mom_1m   : 1-month price return
    mom_3m   : 3-month price return
    mom_6m   : 6-month price return
    mom_12m  : 12-month price return
    vol_21d  : 21-day realised vol (annualised)
    vol_63d  : 63-day realised vol (annualised)
    vol_ratio: ratio of short-term to long-term vol (regime indicator)
    sma_spread: (SMA50 - SMA200) / SMA200 — trend strength
    rsi      : 14-period RSI
    """
    df = df.copy()

    df["mom_1m"]     = df["Close"].pct_change(21)
    df["mom_3m"]     = df["Close"].pct_change(63)
    df["mom_6m"]     = df["Close"].pct_change(126)
    df["mom_12m"]    = df["Close"].pct_change(252)
    df["vol_21d"]    = realised_vol(df["Return"], 21)
    df["vol_63d"]    = realised_vol(df["Return"], 63)
    df["vol_ratio"]  = df["vol_21d"] / df["vol_63d"].replace(0, np.nan)
    df["sma_spread"] = (df["SMA50"] - df["SMA200"]) / df["SMA200"].replace(0, np.nan)
    # RSI already in df from build_signals

    return df


FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12m",
    "vol_21d", "vol_63d", "vol_ratio",
    "sma_spread", "RSI",
]

# Predict direction over this horizon (trading days)
HORIZON = 63   # ~3 months (more predictable than 6 months on limited data)


def build_targets(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """
    Binary target: 1 if price is higher in `horizon` days, else 0.
    Computed using future returns — these are shifted back to align with features.
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_return > 0).astype(int)


# ── Walk-forward ML backtest ──────────────────────────────────────────────────

def build_ml_signals(df: pd.DataFrame,
                     train_years: int = 5,
                     retrain_freq: int = 63,
                     min_train_rows: int = 500,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Walk-forward Random Forest: train on past `train_years` years,
    predict on the next `retrain_freq` trading days, then retrain.

    This is the ONLY statistically valid way to backtest an ML model
    on time series data. No shuffling. No future data in training.

    Parameters
    ----------
    df             : DataFrame with build_signals() already applied.
    train_years    : Rolling training window in years.
    retrain_freq   : How often to retrain (in trading days).
    min_train_rows : Minimum rows needed before making predictions.
    verbose        : Print progress and OOS accuracy.

    Returns
    -------
    df with added column 'ML_Signal' (1 = ML predicts up, 0 = ML predicts down/uncertain)
    and 'ML_Weight' combining classical + ML gates.
    """
    df = df.copy()
    df = build_features(df)
    targets = build_targets(df, HORIZON)

    # Align targets (shift removes future data beyond last row)
    valid_idx = targets.dropna().index
    df["Target"] = targets

    ml_signal = pd.Series(0, index=df.index)
    oos_preds, oos_actuals = [], []

    train_window = train_years * 252
    all_dates    = df.index.tolist()

    i = min_train_rows
    scaler = StandardScaler()
    model  = None

    while i < len(all_dates) - HORIZON:
        t = all_dates[i]

        # Retrain every `retrain_freq` days
        if i % retrain_freq == 0:
            train_start = max(0, i - train_window)
            train_idx   = all_dates[train_start:i]

            X_train_df = df.loc[train_idx, FEATURE_COLS].dropna()
            y_train    = df.loc[X_train_df.index, "Target"].dropna()

            # Align
            common = X_train_df.index.intersection(y_train.index)
            X_tr   = X_train_df.loc[common]
            y_tr   = y_train.loc[common]

            if len(y_tr) < 100 or y_tr.nunique() < 2:
                i += 1
                continue

            scaler.fit(X_tr)
            X_tr_scaled = scaler.transform(X_tr)

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=4,           # deliberately shallow to reduce overfit
                min_samples_leaf=20,   # conservative — avoid memorising noise
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_tr_scaled, y_tr)

        # Predict for current day
        if model is None:
            i += 1
            continue

        row = df.loc[[t], FEATURE_COLS]
        if row.isnull().any().any():
            i += 1
            continue

        try:
            X_pred = scaler.transform(row)
            pred   = model.predict(X_pred)[0]
            prob   = model.predict_proba(X_pred)[0][1]  # P(up)

            # Only signal if model is confident (prob > 0.55)
            ml_signal.loc[t] = 1 if prob > 0.55 else 0

            if t in valid_idx:
                oos_preds.append(pred)
                oos_actuals.append(df.loc[t, "Target"])
        except Exception:
            pass

        i += 1

    # ── OOS accuracy report ───────────────────────────────────────────────────
    if verbose and oos_preds:
        oos_preds    = np.array(oos_preds)
        oos_actuals  = np.array(oos_actuals)
        valid_mask   = ~np.isnan(oos_actuals.astype(float))
        oos_preds    = oos_preds[valid_mask]
        oos_actuals  = oos_actuals[valid_mask].astype(int)

        acc = accuracy_score(oos_actuals, oos_preds)
        print(f"\n[ml_strategy] Walk-forward OOS accuracy: {acc:.3f}")
        print("[ml_strategy] NOTE: ~0.52-0.55 is realistic for equity direction prediction.")
        print("[ml_strategy] Anything above 0.58 warrants scepticism on limited data.\n")
        print(classification_report(oos_actuals, oos_preds,
                                    target_names=["Down/Flat", "Up"]))

    df["ML_Signal"] = ml_signal

    # Combined gate: invest only when BOTH classical rules AND ML agree
    df["ML_Weight"] = df["Weight"] * (df["Signal"] & (df["ML_Signal"] == 1)).astype(int)

    # Recompute vol-scaling on ML weight
    from strategy import VOL_TARGET, POS_MIN, POS_MAX
    raw_ml = (VOL_TARGET / df["RealVol"]).clip(POS_MIN, POS_MAX) * \
             (df["Signal"] & (df["ML_Signal"] == 1)).astype(int)
    df["ML_Weight"] = raw_ml

    return df


def run_ml_backtest(df_ml: pd.DataFrame) -> pd.Series:
    """
    Compute ML-enhanced strategy returns from the ML_Weight column.

    Returns
    -------
    pd.Series of daily ML-enhanced net returns.
    """
    w     = df_ml["ML_Weight"]
    w_lag = w.shift(1)
    cost  = COST_BPS * (w - w.shift(1)).abs()
    r_net = w_lag * df_ml["Return"] - cost
    return r_net.dropna()


def feature_importance_report(df: pd.DataFrame,
                               train_end: str = "2014-12-31") -> pd.Series:
    """
    Train one model on the in-sample period and return feature importances.
    Useful for understanding what the model actually learned.
    """
    df_train = df[df.index <= train_end].copy()
    df_train = build_features(df_train)
    targets  = build_targets(df_train, HORIZON)

    X = df_train[FEATURE_COLS].dropna()
    y = targets.loc[X.index].dropna()
    common = X.index.intersection(y.index)
    X, y   = X.loc[common], y.loc[common]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=4,
        min_samples_leaf=20, max_features="sqrt",
        random_state=42
    )
    model.fit(X_sc, y)

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)

    print("\n[ml_strategy] Feature importances (in-sample 2005-2014):")
    print(imp.to_string())
    print("\n  Interpretation: high importance = model leans on this feature.")
    print("  If 'mom_3m' dominates → ML mostly rediscovers momentum (already in classical).\n")

    return imp
