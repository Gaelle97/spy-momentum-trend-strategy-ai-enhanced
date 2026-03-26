"""
generate_analysis.py
====================
Generate all figures for the research paper and README.

Figures produced
----------------
  01_wealth_curves.png        — Cumulative wealth: Strategy vs Buy&Hold vs 60/40 vs Vol-Only
  02_drawdown_profile.png     — Drawdown comparison
  03_rolling_sharpe.png       — 252-day rolling Sharpe ratio
  04_return_distribution.png  — Return distribution with fat-tail annotation
  05_monte_carlo.png          — Monte Carlo simulation fan chart
  06_walk_forward.png         — In-sample vs Out-of-sample Sharpe bar chart
  07_rsi_ablation.png         — With vs Without RSI comparison
  08_benchmark_table.png      — Clean metrics table as figure
  dashboard.png               — 2x3 summary dashboard

Usage
-----
    python generate_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import norm

from data_loader import load_spy
from strategy    import build_signals, compute_returns
from backtest    import (
    performance_metrics, drawdown_series,
    simple_vol_targeting, sixty_forty,
    walk_forward_backtest, rsi_ablation,
    run_full_backtest,
)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Design system ─────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.5,
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    8.5,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.frameon":    False,
}
plt.rcParams.update(STYLE)

C_STRAT = "#1B3A5C"   # navy — classical strategy
C_BH    = "#C8973A"   # gold  — buy & hold
C_6040  = "#6B7280"   # gray  — 60/40
C_VT    = "#0F6B5C"   # teal  — vol-targeting only
C_ML    = "#7C3AED"   # purple — ML-enhanced
C_GRID  = "#E5E7EB"

ALPHA_FILL = 0.12


def _save(fig, name: str, dpi: int = 150):
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 01 — Cumulative wealth ─────────────────────────────────────────────────────

def plot_wealth(df_strat, r_6040, r_vt):
    fig, ax = plt.subplots(figsize=(9, 4))

    w_strat = (1 + df_strat["StratReturn"]).cumprod()
    w_bh    = (1 + df_strat["BHReturn"]).cumprod()
    w_6040  = (1 + r_6040).cumprod()
    w_vt    = (1 + r_vt).cumprod()

    ax.plot(w_bh,    color=C_BH,    lw=1.5, label="Buy & Hold SPY")
    ax.plot(w_strat, color=C_STRAT, lw=1.8, label="Classical Strategy")
    ax.plot(w_6040,  color=C_6040,  lw=1.2, ls="--", label="60/40 Benchmark")
    ax.plot(w_vt,    color=C_VT,    lw=1.2, ls=":",  label="Vol-Targeting Only (no signals)")

    ax.set_title("Cumulative Wealth — $1 Invested in 2005")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("")
    ax.legend(loc="upper left")

    # Annotate final values
    for w, c, label in [(w_strat, C_STRAT, "Strategy"), (w_bh, C_BH, "B&H")]:
        ax.annotate(f"${w.iloc[-1]:.1f}", xy=(w.index[-1], w.iloc[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    color=c, fontsize=8, va="center")

    ax.text(0.01, 0.02,
            "NOTE: Strategy earns less — by design.\n"
            "Its value is drawdown control, not absolute return.",
            transform=ax.transAxes, fontsize=7.5,
            color="#6B7280", fontstyle="italic")

    fig.tight_layout()
    _save(fig, "01_wealth_curves.png")


# ── 02 — Drawdown ─────────────────────────────────────────────────────────────

def plot_drawdown(df_strat, r_6040, r_vt):
    fig, ax = plt.subplots(figsize=(9, 3.5))

    dd_s  = drawdown_series(df_strat["StratReturn"]) * 100
    dd_bh = drawdown_series(df_strat["BHReturn"])    * 100
    dd_60 = drawdown_series(r_6040)                  * 100

    ax.fill_between(dd_bh.index, dd_bh, 0, color=C_BH,    alpha=ALPHA_FILL)
    ax.fill_between(dd_s.index,  dd_s,  0, color=C_STRAT,  alpha=0.25)

    ax.plot(dd_bh, color=C_BH,    lw=1.2, label=f"Buy & Hold  (MDD {dd_bh.min():.1f}%)")
    ax.plot(dd_s,  color=C_STRAT, lw=1.6, label=f"Strategy    (MDD {dd_s.min():.1f}%)")
    ax.plot(dd_60, color=C_6040,  lw=1.0, ls="--", label=f"60/40     (MDD {dd_60.min():.1f}%)")

    ax.set_title("Drawdown Profile — Downside Risk Comparison")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.set_ylim(top=2)

    # Mark major crises
    for yr, label in [("2009-03-09", "GFC\n2008–09"),
                       ("2020-03-23", "COVID\n2020"),
                       ("2022-10-12", "Rate hike\n2022")]:
        try:
            x = pd.Timestamp(yr)
            if x in dd_bh.index or dd_bh.index.asof(x) is not None:
                ax.axvline(x, color="#D1D5DB", lw=0.8, ls="--", zorder=0)
                ax.text(x, ax.get_ylim()[0] * 0.9, label,
                        fontsize=7, ha="center", color="#9CA3AF")
        except Exception:
            pass

    fig.tight_layout()
    _save(fig, "02_drawdown_profile.png")


# ── 03 — Rolling Sharpe ───────────────────────────────────────────────────────

def plot_rolling_sharpe(df_strat):
    fig, ax = plt.subplots(figsize=(9, 3))

    roll_s  = df_strat["StratReturn"].rolling(252).apply(
        lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan)
    roll_bh = df_strat["BHReturn"].rolling(252).apply(
        lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan)

    ax.plot(roll_bh, color=C_BH,    lw=1.2, alpha=0.8, label="Buy & Hold")
    ax.plot(roll_s,  color=C_STRAT, lw=1.6, label="Strategy")
    ax.axhline(0, color="#111111", lw=0.6)
    ax.axhline(1, color=C_STRAT, lw=0.5, ls=":", alpha=0.5)
    ax.text(ax.get_xlim()[0], 1.05, "Sharpe = 1 (excellent)", fontsize=7, color=C_STRAT)

    ax.set_title("Rolling 12-Month Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend()
    ax.set_ylim(-3, 3.5)

    fig.tight_layout()
    _save(fig, "03_rolling_sharpe.png")


# ── 04 — Return distribution ──────────────────────────────────────────────────

def plot_return_dist(df_strat):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    r_s  = df_strat["StratReturn"].dropna() * 100
    r_bh = df_strat["BHReturn"].dropna()    * 100

    ax.hist(r_bh, bins=100, color=C_BH,    alpha=0.45, density=True, label="Buy & Hold")
    ax.hist(r_s,  bins=100, color=C_STRAT, alpha=0.55, density=True, label="Strategy")

    # Normal fit
    mu, std = r_s.mean(), r_s.std()
    x = np.linspace(r_s.min(), r_s.max(), 300)
    ax.plot(x, norm.pdf(x, mu, std), color="#111111", lw=1.2, ls="--",
            label="Normal fit (strategy)")

    skew = r_s.skew()
    kurt = r_s.kurtosis()
    ax.text(0.97, 0.95, f"Skew={skew:.2f}\nKurt={kurt:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=C_STRAT)
    ax.text(0.97, 0.65,
            "! Excess kurtosis: fat tails.\nStandard VaR underestimates risk.",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#8B1A1A", fontstyle="italic")

    ax.set_title("Daily Return Distribution — Fat Tails vs Normal")
    ax.set_xlabel("Daily Return (%)")
    ax.legend()

    fig.tight_layout()
    _save(fig, "04_return_distribution.png")


# ── 05 — Monte Carlo ──────────────────────────────────────────────────────────

def plot_monte_carlo(df_strat, n_paths: int = 1000, horizon: int = 252):
    fig, ax = plt.subplots(figsize=(7, 4))

    r_s  = df_strat["StratReturn"].dropna()
    mu   = r_s.mean()
    std  = r_s.std()

    rng    = np.random.default_rng(42)
    paths  = rng.normal(mu, std, size=(n_paths, horizon))
    wealth = np.cumprod(1 + paths, axis=1)

    pct = np.percentile(wealth, [5, 25, 50, 75, 95], axis=0)
    x   = np.arange(horizon)

    ax.fill_between(x, pct[0], pct[4], alpha=0.12, color=C_STRAT, label="5–95th pct")
    ax.fill_between(x, pct[1], pct[3], alpha=0.25, color=C_STRAT, label="25–75th pct")
    ax.plot(x, pct[2], color=C_STRAT, lw=2, label=f"Median ({pct[2,-1]:.0%})")
    ax.axhline(1, color="#D1D5DB", lw=0.8)

    ax.set_title(f"Monte Carlo Simulation — {n_paths:,} Paths × 1 Year")
    ax.set_xlabel("Trading Days Ahead")
    ax.set_ylabel("Wealth (start = 1.0)")
    ax.legend()

    ax.text(0.02, 0.05,
            f"5th pct: {pct[0,-1]:.0%}  |  Median: {pct[2,-1]:.0%}  |  95th pct: {pct[4,-1]:.0%}",
            transform=ax.transAxes, fontsize=8, color="#6B7280")

    fig.tight_layout()
    _save(fig, "05_monte_carlo.png")


# ── 06 — Walk-forward ─────────────────────────────────────────────────────────

def plot_walk_forward(df_raw):
    wf = walk_forward_backtest(df_raw)
    labels  = list(wf.keys())
    sharpes = [wf[l]["Sharpe Ratio"] for l in labels]
    caldars = [wf[l]["Calmar Ratio"] for l in labels]
    cagrs   = [wf[l]["CAGR (%)"]     for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    colors = [C_STRAT, "#0F6B5C"]

    for ax, vals, title in zip(axes,
                                [sharpes, caldars, cagrs],
                                ["Sharpe Ratio", "Calmar Ratio", "CAGR (%)"]):
        bars = ax.bar(["In-sample\n2005–2014", "Out-of-sample\n2015–2024"],
                      vals, color=colors, width=0.5)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.35)

    axes[0].text(0.5, -0.28,
                 "Key: OOS metrics close to IS metrics → strategy is robust, not overfit.",
                 transform=axes[0].transAxes, ha="center", fontsize=7.5,
                 color="#6B7280", fontstyle="italic")

    fig.suptitle("Walk-Forward Validation — In-Sample vs Out-of-Sample", y=1.02)
    fig.tight_layout()
    _save(fig, "06_walk_forward.png")


# ── 07 — RSI ablation ─────────────────────────────────────────────────────────

def plot_rsi_ablation(df_raw):
    tbl  = rsi_ablation(df_raw)
    metrics = ["Sharpe Ratio", "Calmar Ratio", "Max Drawdown (%)", "CAGR (%)"]

    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    colors    = [C_STRAT, C_VT]
    labels    = ["With RSI", "Without RSI"]

    for ax, met in zip(axes, metrics):
        vals = [tbl.loc[l, met] for l in tbl.index]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        ax.set_title(met)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height())*0.02,
                    f"{v:.2f}", ha="center", fontsize=8.5)

    axes[0].text(0.5, -0.30,
                 "If 'Without RSI' metrics >= 'With RSI' → RSI adds noise, not edge.",
                 transform=axes[0].transAxes, ha="center", fontsize=7.5,
                 color="#8B1A1A", fontstyle="italic")

    fig.suptitle("RSI Ablation Study — Does the RSI Filter Add Value?", y=1.02)
    fig.tight_layout()
    _save(fig, "07_rsi_ablation.png")


# ── Dashboard (2×3) ───────────────────────────────────────────────────────────

def plot_dashboard(df_strat, r_6040, r_vt, df_raw):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0:2])   # wealth — wide
    ax2 = fig.add_subplot(gs[0, 2])     # metrics table
    ax3 = fig.add_subplot(gs[1, 0])     # drawdown
    ax4 = fig.add_subplot(gs[1, 1])     # rolling sharpe
    ax5 = fig.add_subplot(gs[1, 2])     # monte carlo

    # ── 1. Wealth ──────────────────────────────────────────────────────────
    w_s  = (1 + df_strat["StratReturn"]).cumprod()
    w_bh = (1 + df_strat["BHReturn"]).cumprod()
    w_60 = (1 + r_6040).cumprod()
    ax1.plot(w_bh, color=C_BH,    lw=1.4, label="Buy & Hold")
    ax1.plot(w_s,  color=C_STRAT, lw=1.8, label="Strategy")
    ax1.plot(w_60, color=C_6040,  lw=1.0, ls="--", label="60/40")
    ax1.set_title("Cumulative Wealth  ($1 invested 2005)")
    ax1.legend(fontsize=7.5)
    ax1.set_ylabel("$")

    # ── 2. Metrics table ──────────────────────────────────────────────────
    ax2.axis("off")
    m_s  = performance_metrics(df_strat["StratReturn"], "Strategy")
    m_bh = performance_metrics(df_strat["BHReturn"],    "B&H")
    rows = [["Metric", "Strategy", "Buy & Hold"]]
    for met in ["CAGR (%)", "Ann. Vol (%)", "Sharpe Ratio",
                "Max Drawdown (%)", "Calmar Ratio"]:
        rows.append([met, str(m_s[met]), str(m_bh[met])])

    tbl_art = ax2.table(cellText=rows[1:], colLabels=rows[0],
                         cellLoc="center", loc="center")
    tbl_art.auto_set_font_size(False)
    tbl_art.set_fontsize(8)
    tbl_art.scale(1, 1.5)
    for (r, c), cell in tbl_art.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        if r == 0:
            cell.set_facecolor("#1B3A5C")
            cell.set_text_props(color="white", fontsize=8, fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F9FAFB")
    ax2.set_title("Performance Summary", pad=12)

    # ── 3. Drawdown ───────────────────────────────────────────────────────
    dd_s  = drawdown_series(df_strat["StratReturn"]) * 100
    dd_bh = drawdown_series(df_strat["BHReturn"])    * 100
    ax3.fill_between(dd_bh.index, dd_bh, 0, color=C_BH,    alpha=ALPHA_FILL)
    ax3.fill_between(dd_s.index,  dd_s,  0, color=C_STRAT,  alpha=0.2)
    ax3.plot(dd_bh, color=C_BH,    lw=1.0, label=f"B&H MDD {dd_bh.min():.0f}%")
    ax3.plot(dd_s,  color=C_STRAT, lw=1.4, label=f"Strat MDD {dd_s.min():.0f}%")
    ax3.set_title("Drawdown Profile")
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend(fontsize=7)
    ax3.set_ylim(top=2)

    # ── 4. Rolling Sharpe ─────────────────────────────────────────────────
    roll_s = df_strat["StratReturn"].rolling(252).apply(
        lambda r: r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan)
    ax4.plot(roll_s, color=C_STRAT, lw=1.4)
    ax4.axhline(0, color="#111111", lw=0.6)
    ax4.set_title("Rolling 12M Sharpe")
    ax4.set_ylabel("Sharpe")

    # ── 5. Monte Carlo ────────────────────────────────────────────────────
    r_s   = df_strat["StratReturn"].dropna()
    mu, std = r_s.mean(), r_s.std()
    rng   = np.random.default_rng(42)
    paths = rng.normal(mu, std, size=(1000, 252))
    w_mc  = np.cumprod(1 + paths, axis=1)
    pct   = np.percentile(w_mc, [5, 25, 50, 75, 95], axis=0)
    x     = np.arange(252)
    ax5.fill_between(x, pct[0], pct[4], alpha=0.12, color=C_STRAT)
    ax5.fill_between(x, pct[1], pct[3], alpha=0.25, color=C_STRAT)
    ax5.plot(x, pct[2], color=C_STRAT, lw=2, label=f"Median {pct[2,-1]:.0%}")
    ax5.axhline(1, color="#D1D5DB", lw=0.8)
    ax5.set_title("Monte Carlo — 1,000 paths × 1Y")
    ax5.set_xlabel("Trading Days")
    ax5.set_ylabel("Wealth")
    ax5.legend(fontsize=7.5)

    fig.suptitle(
        "Momentum & Trend-Following Strategy  ·  SPY  ·  2005–2024  ·  Vol-Targeted",
        fontsize=11, fontweight="bold", y=1.01)

    _save(fig, "dashboard.png", dpi=180)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[generate_analysis] Loading data...")
    df, is_real = load_spy()

    df_strat = build_signals(df)
    df_strat = compute_returns(df_strat)
    r_6040   = sixty_forty(df).reindex(df_strat.index)
    r_vt     = simple_vol_targeting(df).reindex(df_strat.index)

    print("[generate_analysis] Generating figures...")
    plot_wealth(df_strat, r_6040, r_vt)
    plot_drawdown(df_strat, r_6040, r_vt)
    plot_rolling_sharpe(df_strat)
    plot_return_dist(df_strat)
    plot_monte_carlo(df_strat)
    plot_walk_forward(df)
    plot_rsi_ablation(df)
    plot_dashboard(df_strat, r_6040, r_vt, df)

    print(f"\n[generate_analysis] All figures saved to {FIGURES_DIR.resolve()}")

    if not is_real:
        print("\nWARNING: All figures above use SYNTHETIC data.")
        print("   Re-run with real SPY data for publication-quality results.")


if __name__ == "__main__":
    main()
