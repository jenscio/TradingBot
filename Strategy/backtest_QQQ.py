# backtest_sp500.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtesting import Backtest
from pathlib import Path
from strategy import ARSIstrat


# ---------- Config ----------
START = '2014-12-17'
CUTOFF = "2022-01-02"  
SPREAD = 0.000025       
BH_LAG_DAYS = 40

ASSET = {
    "QQQ":   os.path.join("CSV_files", "BATS_QQQ, 60_a45be.csv")
}

# picked params
PARAMS = {'sl_pct': 0.00521, 'n_fast': 14, 'n_slow': 49, 'n_vslow': 65, 'sig_len': 10}

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

    # Anchor both curves Buy Hold + Strategy to start BH_LAG_DAYS later
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

    # Buy & Hold return computed on the same clipped window
    if len(common_idx) >= 2:
        bh_pct = float((bh_clip.loc[common_idx[-1]] / bh_clip.loc[common_idx[0]] - 1.0) * 100.0)
    else:
        bh_pct = float('nan')

    out = {
    'Sharpe':           float(stats.get('Sharpe Ratio', np.nan)),
    'Sortino':          float(stats.get('Sortino Ratio', np.nan)),
    'Calmar':           float(stats.get('Calmar Ratio', np.nan)),
    'Return %':         float(stats.get('Return [%]', np.nan)),
    'Ann. Return %':    float(stats.get('Return (Ann.) [%]', np.nan)),
    'Max DD %':         float(stats.get('Max. Drawdown [%]', np.nan)),
    'Win Rate %':       float(stats.get('Win Rate [%]', np.nan)),
    'Trades':           int(stats.get('# Trades', stats.get('Trades', np.nan))),
    'Exposure %':       float(stats.get('Exposure Time [%]', np.nan)),
    'Buy&Hold %':       bh_pct,   
}
    return out, panel

def plot_panel(panel, metrics, out_path="graphs/qqq_comparison.png"):
    plt.figure(figsize=(10,6))
    plt.plot(panel.index, panel['Strategy'], label='Strategy', linewidth=1.4)
    plt.plot(panel.index, panel['Buy&Hold'], label='Buy & Hold', linewidth=1.4)
    plt.title(f"QQQ  (Sharpe {metrics['Sharpe']:.2f})")
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, frameon=False, loc='upper left')
    Path(out_path).parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison figure to {out_path}")

def main():
    name, path = list(ASSET.items())[0]
    if not os.path.exists(path):
        print(f"[ERROR] Missing CSV for {name}: {path}")
        return

    df = load_csv_ohlcv(path)
    if df.empty:
        print(f"[WARN] No rows for {name} in requested window")
        return

    res, panel = run_one(df, PARAMS, spread=SPREAD, bh_lag_days=BH_LAG_DAYS)
    res['Asset'] = name

    out = pd.DataFrame([res]).set_index('Asset').round(3)
    print("\nParams used:", PARAMS)
    print("\nBacktest result for QQQ:")
    print(out.to_string())

    Path("CSV_files").mkdir(parents=True, exist_ok=True)
    out_path = Path("CSV_files") / "backtest_results_qqq.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path.resolve()}")

    plot_panel(panel, res)

if __name__ == "__main__":
    main()
