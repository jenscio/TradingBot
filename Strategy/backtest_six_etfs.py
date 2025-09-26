# backtest_six_assets.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtesting import Backtest
from pathlib import Path
from strategy import ARSIstrat

# ---------- Config ----------
START = '2023-01-01'
CUTOFF = "2025-07-17"  
SPREAD = 0.000025   
BH_LAG_DAYS = 40   

print("Current working directory:", os.getcwd())
print("Script folder:", Path(__file__).parent)
csv_path = Path(__file__).parent / "CSV_files" / "BATS_QQQ, 60_a45be.csv"
print("Expected CSV path:", csv_path)
print("Exists?", csv_path.exists())

ASSETS = {
    "QQQ":   os.path.join("CSV_files", "BATS_QQQ, 60_a45be.csv"),
    "MSCI":  os.path.join("CSV_files", "EURONEXT_DLY_IWDA, 60_6c01f.csv"),
    "SMI":   os.path.join("CSV_files", "SIX_DLY_SMI, 60_2d252.csv"),
    "SPX":   os.path.join("CSV_files", "SP_SPX, 60_47660.csv"),
    "CAC40": os.path.join("CSV_files", "TVC_CAC40, 60_aae3c.csv"),
    "DAX":   os.path.join("CSV_files", "XETR_DLY_DAX, 60_dc96e.csv"),
}

# picked params
PARAMS = {'sl_pct': 0.00521, 'n_fast': 14, 'n_slow': 49, 'n_vslow': 65, 'sig_len': 10}
PARAMS = {'sl_pct': 0.00421, 'n_fast': 9, 'n_slow': 20, 'n_vslow': 84, 'sig_len': 7}

# ---------- Helpers ----------
def load_csv_ohlcv(path, start=START, end_excl=CUTOFF):
    df = pd.read_csv(path)

    # Convert time into epoch seconds and sort the df by time
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s', utc=True)
    df = df.set_index('time').sort_index()
    df = df.tz_convert(None)  

    df = df.loc[(df.index >= start) & (df.index < end_excl)].copy()

    # Unify correct column names
    rename_map = {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}
    df = df.rename(columns=rename_map)

    # Keep only the needed columns, ensure float
    need = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[need].astype(float)
    df = df.dropna(subset=['Open','High','Low','Close'])
    return df


def make_strat_with_params(p):
    
    class Tuned(ARSIstrat):
        sl_pct = float(p['sl_pct'])
        n_fast = int(p['n_fast'])
        n_slow = int(p['n_slow'])
        n_vslow = int(p['n_vslow'])
        sig_len = int(p['sig_len'])
    return Tuned


def run_one(df, params, spread=SPREAD, bh_lag_days=BH_LAG_DAYS):
    Strat = make_strat_with_params(params)
    bt = Backtest(df, Strat, cash=1_000_000, commission=0.0, spread=spread, finalize_trades=True)
    stats = bt.run()

    # Strategy equity in absolute currency
    eq = stats._equity_curve['Equity'].astype(float).copy()

    # Align equity index to price index
    if len(eq) == len(df):
        eq.index = df.index
    else:
        eq.index = df.index[-len(eq):]

    # Anchor both curves Buy and Hold + Strategy to start BH_LAG_DAYS later
    anchor = df.index[0] + pd.Timedelta(days=bh_lag_days)

    # Clip to anchor date
    eq_clip = eq.loc[eq.index >= anchor]
    bh_clip = df['Close'].astype(float).loc[df.index >= anchor]

    
    common_idx = eq_clip.index.intersection(bh_clip.index).sort_values()
    panel = pd.DataFrame({
        'Strategy': eq_clip.loc[common_idx],
        'Buy&Hold': bh_clip.loc[common_idx]
    }).dropna()
    panel = panel / panel.iloc[0]

    # Buy and Hold return computed on the same clipped window
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
        'Buy&Hold %':       bh_pct,   
    }
    return out, panel


def plot_grid(panels_by_asset, metrics_by_asset, out_path="six_assets_comparison.png"):
    # Make a 3x2 grid comparing Strategy vs Buy&Hold for each asset.
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
    print("\nBacktest results across 6 etf's (sorted by Sharpe):")
    print(out.to_string())

    # Save to csv for record
    csv_dir = Path("CSV_files")
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = csv_dir / "backtest_results_6_assets.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path.resolve()}")

    # Plot grid
    # Keep the asset order consistent  (sorted by Sharpe)
    ordered_panels = {name: panels[name] for name in out.index if name in panels}
    ordered_metrics = {name: metrics_map[name] for name in out.index if name in metrics_map}
    Path("graphs").mkdir(exist_ok=True)
    plot_grid(ordered_panels, ordered_metrics,
          out_path="graphs/six_assets_comparison.png")


if __name__ == "__main__":
    main()
