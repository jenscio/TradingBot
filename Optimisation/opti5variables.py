# optimize_bt_optimize.py
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import multiprocessing as mp
import backtesting as btng
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from Strategy.strategy import ARSIstrat

# Sl grid: values from 0.00010 to 0.02000 in 0.00001 increments
SL_GRID = [round(x, 5) for x in np.linspace(0.0001, 0.0200, int((0.0200 - 0.0001) / 0.00001) + 1)]

# ---------- Data Loading ----------
CSV_PATH = "CSV_files/BATS_QQQ, 60_a45be.csv"
START = "2014-12-17"
END_EXCL = "2022-01-01"

# Load CSV
full_df = pd.read_csv(CSV_PATH)
# Parse epoch time to datetime
full_df['time'] = pd.to_datetime(full_df['time'].astype(int), unit='s', utc =True)
# Set time as index and sort
full_df.set_index('time', inplace=True)
full_df = full_df.tz_convert(None)  
full_df.sort_index(inplace=True)
# Unify correct column names
full_df = full_df.loc[(full_df.index >= START) & (full_df.index < END_EXCL)].copy()
full_df = full_df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
full_df = full_df[['Open','High','Low','Close','Volume']].astype(float)
# Keep only the needed columns, ensure float
full_df = full_df.dropna(subset=['Open','High','Low','Close'])
print(full_df.head())

# ---------- Backtest + optimiser config ----------
btng.Pool = mp.get_context('spawn').Pool 

bt = Backtest(
    full_df,
    ARSIstrat,
    cash=1_000_000,
    commission=0.0,
    spread=0.000025,
    finalize_trades=True
)

# ---------- Constraints ----------
MIN_GAP_FS = 3   # n_slow - n_fast  >= 3
MIN_GAP_SV = 7   # n_vslow - n_slow >= 7

# Constraints: enforce parameter ordering &andminimum gaps between fast/slow/vslow,
# and require positive signal length and sl
def _order_ok(p):
    return (
        (p.n_slow  - p.n_fast  >= MIN_GAP_FS) and
        (p.n_vslow - p.n_slow  >= MIN_GAP_SV) and
        (p.n_fast  < p.n_slow  < p.n_vslow)   and
        (p.sig_len > 0) and (p.sl_pct > 0)
    )

# ---------- Search space ----------
param_space = dict(
    n_fast  = list(range(5, 26, 1)),
    n_slow  = list(range(10, 56, 1)),
    n_vslow = list(range(30, 121, 1)),
    sig_len = list(range(5, 31, 1)),
    sl_pct  = SL_GRID, # in 0.00001 steps
)

# ---------- Run Optimise ----------
if __name__ == '__main__':
    stats, heatmap = bt.optimize(
        **param_space,
        maximize='Sharpe Ratio',
        return_heatmap=True,
        constraint=_order_ok,
        method='sambo',
        max_tries=5000,
        random_state=81
    )

    print(stats[['Sharpe Ratio', 'Return [%]', '# Trades']])

    # Best params from optimizer; SNAP sl_pct for printing too
    best = stats._strategy
    best_params = {
        'n_fast':  int(best.n_fast),
        'n_slow':  int(best.n_slow),
        'n_vslow': int(best.n_vslow),
        'sig_len': int(best.sig_len),
        'sl_pct':  float(min(SL_GRID, key=lambda g: abs(g - float(best.sl_pct)))),  # Needed because we use sambo, and not grid search
    }
    print("Best params:", best_params)

    # Best by  heatmap (works for both Series and DataFrame)
    if hasattr(heatmap, 'index'):
        if isinstance(heatmap, pd.Series):
            best_idx = heatmap.astype(float).idxmax()
            
            names = heatmap.index.names
            vals  = best_idx if isinstance(best_idx, tuple) else (best_idx,)
            best_params_hm = dict(zip(names, vals))
        else:  # DataFrame
            col = next(c for c in heatmap.columns if str(c).lower().startswith('sharpe'))
            best_idx = heatmap[col].astype(float).idxmax()
            names = heatmap.index.names
            best_params_hm = dict(zip(names, best_idx if isinstance(best_idx, tuple) else (best_idx,)))

        if 'sl_pct' in best_params_hm:
            best_params_hm['sl_pct'] = float(min(SL_GRID, key=lambda g: abs(g - float(best_params_hm['sl_pct']))))
        print("Best (from heatmap):", best_params_hm)

        # ---------- Top 25 unique parameter sets ----------
        seen = set()  # to track printed combos
        if isinstance(heatmap, pd.Series):
            top25 = heatmap.sort_values(ascending=False).head(100)  
            print("\nTop 25 unique parameter sets:")
            count = 0
            for idx, val in top25.items():
                params_dict = dict(zip(heatmap.index.names, idx if isinstance(idx, tuple) else (idx,)))
                if 'sl_pct' in params_dict:
                    params_dict['sl_pct'] = float(min(SL_GRID, key=lambda g: abs(g - float(params_dict['sl_pct']))))
                params_tuple = tuple(params_dict.items())
                if params_tuple not in seen:
                    seen.add(params_tuple)
                    print(f"Sharpe {float(val):.5f}  ->  {params_dict}")
                    count += 1
                if count >= 25:
                    break
        else:  # DataFrame
            col = next(c for c in heatmap.columns if str(c).lower().startswith('sharpe'))
            top25 = heatmap.sort_values(by=col, ascending=False).head(100)
            print("\nTop 25 unique parameter sets:")
            count = 0
            for idx, row in top25.iterrows():
                params_dict = dict(zip(heatmap.index.names, idx if isinstance(idx, tuple) else (idx,)))
                if 'sl_pct' in params_dict:
                    params_dict['sl_pct'] = float(min(SL_GRID, key=lambda g: abs(g - float(params_dict['sl_pct']))))
                params_tuple = tuple(params_dict.items())
                if params_tuple not in seen:
                    seen.add(params_tuple)
                    print(f"Sharpe {float(row[col]):.5f}  ->  {params_dict}")
                    count += 1
                if count >= 25:
                    break
