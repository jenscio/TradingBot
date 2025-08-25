# backtest_six_assets.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtesting import Backtest

# --- import your strategy class ---
try:
    from tradingbot.strategies.strategy import ARSIstrat
except Exception as e:
    raise SystemExit(
        "Could not import ARSIstrat from tradingbot/strategies/strategy.py.\n"
        "Make sure the file is import-safe (no CSV reading at import time).\n"
        f"Original error: {e}"
    )

# ---------- Config ----------
CUTOFF = "2025-07-17"  # end-exclusive cutoff for data
SPREAD = 0.0001        # tweak per market if you like
BH_LAG_DAYS = 30

ASSETS = {
    "QQQ":   os.path.join("data", "BATS_QQQ, 60_a45be.csv"),
    "MSCI":  os.path.join("data", "EURONEXT_DLY_IWDA, 60_6c01f.csv"),
    "SMI":   os.path.join("data", "SIX_DLY_SMI, 60_2d252.csv"),
    "SPX":   os.path.join("data", "SP_SPX, 60_c5754.csv"),
    "CAC40": os.path.join("data", "TVC_CAC40, 60_aae3c.csv"),
    "DAX":   os.path.join("data", "XETR_DLY_DAX, 60_dc96e.csv"),
}

# Your picked params
PARAMS = {'sl_pct': 0.0031, 'n_fast': 24, 'n_slow': 37, 'n_vslow': 105, 'sig_len': 29}

# ---------- Helpers ----------
def load_csv_ohlcv(path, start='2018-01-02', end_excl=CUTOFF):
    """Load your CSV schema (epoch-seconds in 'time') to Backtesting.py-friendly OHLCV."""
    df = pd.read_csv(path)

    if 'time' not in df.columns:
        raise ValueError(f"'time' column not found in {path}")

    # time is epoch seconds
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
    df = df.set_index('time').sort_index()
    df = df.tz_convert(None)  # make tz-naive for Backtesting.py

    df = df.loc[(df.index >= start) & (df.index < end_excl)].copy()

    # unify column names
    rename_map = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}
    df = df.rename(columns=rename_map)

    # keep only needed cols, ensure float
    need = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[need].astype(float)
    df['Volume'] = df['Volume'].fillna(0.0)
    df = df.dropna(subset=['Open','High','Low','Close'])
    return df


def make_strat_with_params(p):
    """Bind params to ARSIstrat without touching its code."""
    class Tuned(ARSIstrat):
        sl_pct = float(p['sl_pct'])
        n_fast = int(p['n_fast'])
        n_slow = int(p['n_slow'])
        n_vslow = int(p['n_vslow'])
        sig_len = int(p['sig_len'])
    return Tuned


def run_one(df, params, spread=SPREAD, bh_lag_days=0):
    Strat = make_strat_with_params(params)
    bt = Backtest(df, Strat, cash=1_000_000, commission=0.0, spread=spread, finalize_trades=True)
    stats = bt.run()

    # Strategy equity in absolute currency → normalize later
    eq = stats._equity_curve['Equity'].astype(float).copy()

    # Align equity index to price index
    if len(eq) == len(df):
        eq.index = df.index
    else:
        eq.index = df.index[-len(eq):]

    # --- anchor both curves BH + Strategy to start bh_lag_days later ---
    anchor = df.index[0] + pd.Timedelta(days=bh_lag_days)

    # clip to anchor date
    eq_clip = eq.loc[eq.index >= anchor]
    bh_clip = df['Close'].astype(float).loc[df.index >= anchor]

    # common timeline, then rebase both to 1 at first point
    common_idx = eq_clip.index.intersection(bh_clip.index).sort_values()
    panel = pd.DataFrame({
        'Strategy': eq_clip.loc[common_idx],
        'Buy&Hold': bh_clip.loc[common_idx]
    }).dropna()
    panel = panel / panel.iloc[0]

    # Buy & Hold return computed on the same clipped window
    if len(common_idx) >= 2:
        bh_pct = float((bh_clip.loc[common_idx[-1]] / bh_clip.loc[common_idx[0]] - 1.0) * 100.0)
    else:
        bh_pct = float('nan')

    out = {
        'Sharpe':           float(stats.get('Sharpe Ratio', np.nan)),
        'Return %':         float(stats.get('Return [%]', np.nan)),
        'Ann. Return %':    float(stats.get('Return (Ann.) [%]', np.nan)),
        'Max DD %':         float(stats.get('Max. Drawdown [%]', np.nan)),
        'Win Rate %':       float(stats.get('Win Rate [%]', np.nan)),
        'Trades':           int(stats.get('# Trades', stats.get('Trades', np.nan))),
        'Exposure %':       float(stats.get('Exposure Time [%]', np.nan)),
        'Buy&Hold %':       bh_pct,   # << now matches the plotted window
    }
    return out, panel


def plot_grid(panels_by_asset, metrics_by_asset, out_path="six_assets_comparison.png"):
    """Make a 3x2 grid comparing Strategy vs Buy&Hold for each asset."""
    names = list(panels_by_asset.keys())
    rows, cols = 3, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10), sharex=False)
    axes = axes.ravel()

    for i, name in enumerate(names):
        ax = axes[i]
        panel = panels_by_asset[name]
        metr = metrics_by_asset[name]

        if panel.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(name)
            continue

        ax.plot(panel.index, panel['Strategy'], label='Strategy', linewidth=1.4)
        ax.plot(panel.index, panel['Buy&Hold'], label='Buy & Hold', linewidth=1.4)
        ax.set_title(f"{name}  (Sharpe {metr['Sharpe']:.2f})")
        ax.set_ylabel("Growth of $1")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False, loc='upper left')

    # Hide any unused subplots (in case fewer than 6)
    for j in range(len(names), rows*cols):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved comparison figure to {out_path}")


def main():
    rows = []
    panels = {}
    metrics_map = {}

    for name, path in ASSETS.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing CSV for {name}: {path}")
            continue

        try:
            df = load_csv_ohlcv(path)
            if df.empty:
                print(f"[WARN] No rows for {name} in requested window")
                continue
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            continue

        try:
            res, panel = run_one(df, PARAMS, spread=SPREAD, bh_lag_days=BH_LAG_DAYS)
            res['Asset'] = name
            rows.append(res)
            panels[name] = panel
            metrics_map[name] = res
        except Exception as e:
            print(f"[ERROR] Backtest failed for {name}: {e}")

    if not rows:
        print("No results.")
        return

    out = pd.DataFrame(rows).set_index('Asset').round(3)
    out = out.sort_values('Sharpe', ascending=False)

    print("\nParams used:", PARAMS)
    print("\nBacktest results across 6 assets (sorted by Sharpe):")
    print(out.to_string())

    # Save to CSV for record
    out_path = "backtest_results_6_assets.csv"
    out.to_csv(out_path)
    print(f"\nSaved results to {out_path}")

    # Plot grid
    # Keep the asset order consistent with the table (sorted by Sharpe)
    ordered_panels = {name: panels[name] for name in out.index if name in panels}
    ordered_metrics = {name: metrics_map[name] for name in out.index if name in metrics_map}
    plot_grid(ordered_panels, ordered_metrics, out_path="six_assets_comparison.png")


if __name__ == "__main__":
    main()
