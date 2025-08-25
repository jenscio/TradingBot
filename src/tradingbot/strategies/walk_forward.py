import os
import numpy as np
import pandas as pd
import pickle
from backtesting import Backtest,Strategy
from backtesting.lib import resample_apply
import matplotlib.pyplot as plt

from tradingbot.strategies.strategy import ARSIstrat

import sys, inspect
print(ARSIstrat.__module__)                               # should be 'tradingbot.strategies.strategy'
print(sys.modules[ARSIstrat.__module__].__file__)         # path to strategy.py you expect
print([a for a in dir(ARSIstrat) if a.startswith('n_')])  # should show ['n_fast','n_slow','n_vslow']


full_df = pd.read_csv('data/BATS_QQQ, 60_a45be.csv')
start = '2018-01-01'
end_excl = '2024-01-01'   # exclusive upper bound (covers all of 2018–2022)
full_df['time'] = pd.to_datetime(full_df['time'].astype(int), unit='s')
full_df.set_index('time', inplace=True)
full_df.sort_index(inplace=True)
full_df = full_df.loc[(full_df.index >= start) & (full_df.index < end_excl)].copy()
full_df = full_df.rename(columns={
    'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'
})
print(full_df.head())


def walk_forward(
    strategy, 
    data_full,
    warmup_bars,
    lookback_bars, 
    validation_bars, 
    cash = 1_000_000
    ):

    stats_master = []
    for i in range(lookback_bars + warmup_bars, len(data_full) - validation_bars, validation_bars):
        training_data  = data_full.iloc[i-lookback_bars - warmup_bars:i]
        validation_data = data_full.iloc[i-warmup_bars:i+validation_bars]
        bt_tr = Backtest(training_data, strategy, cash=cash, commission=0.0, spread=0.0001, finalize_trades=True)

        # Return the heatmap so we can get params regardless of Stats quirks
        st_tr, heat = bt_tr.optimize(
            n_fast=range(5, 16),
            n_slow=range(10, 41, 2),
            n_vslow=range(40, 121, 5),
            constraint=lambda p: p.n_fast < p.n_slow < p.n_vslow,
            maximize='Sharpe Ratio',
            return_heatmap=True,
        )

        # Derive best param combo from the heatmap
        # heat is usually a pandas.Series with MultiIndex (levels = param names)
        if isinstance(heat, pd.Series):
            best_idx = heat.idxmax()                       # tuple for MultiIndex, scalar otherwise
            names = heat.index.names
            if not isinstance(best_idx, tuple):
                best_idx = (best_idx,)
            params = dict(zip(names, best_idx))
        else:  # fall back if your version returns a DataFrame
            best_idx = heat['Sharpe Ratio'].idxmax()
            if not isinstance(best_idx, tuple):
                best_idx = (best_idx,)
            names = heat.index.names
            params = dict(zip(names, best_idx))

        n_fast  = int(params['n_fast'])
        n_slow  = int(params['n_slow'])
        n_vslow = int(params['n_vslow'])

        bt_validation = Backtest(validation_data, strategy, cash = cash)
        stats_validation = bt_validation.run(n_fast=n_fast,n_slow=n_slow,n_vslow=n_vslow)

        stats_master.append(stats_validation)
    return stats_master


lookback_bars = 2000
validation_bars = 400
warmup_bars = 120

stats = walk_forward(
    ARSIstrat,full_df,
    lookback_bars=lookback_bars,
    validation_bars=validation_bars,
    warmup_bars=warmup_bars)

print(stats)
for fold, s in enumerate(stats, start=1):  # fold = 1,2,3,...
    strat = getattr(s, "_strategy", None) or s.get("_strategy")
    print(f"Fold {fold}: n_fast={strat.n_fast}, n_slow={strat.n_slow}, n_vslow={strat.n_vslow}")

import pandas as pd, numpy as np, re

def extract_params_from_stats(stats_list):
    rows = []
    for s in stats_list:
        # Try to get the live strategy instance (preferred)
        strat = getattr(s, "_strategy", None)
        if strat is None and "_strategy" in s:
            strat = s["_strategy"]

        if strat is not None:
            nf = getattr(strat, "n_fast", np.nan)
            ns = getattr(strat, "n_slow", np.nan)
            nv = getattr(strat, "n_vslow", np.nan)
        else:
            # Fallback: parse the repr if that's all you have
            txt = str(s.get("_strategy", ""))
            m = re.search(r"n_fast=(\d+).*?n_slow=(\d+).*?n_vslow=(\d+)", txt)
            nf, ns, nv = map(int, m.groups()) if m else (np.nan, np.nan, np.nan)

        rows.append({
            "Start": s["Start"],
            "End": s["End"],
            "n_fast": int(nf) if pd.notna(nf) else np.nan,
            "n_slow": int(ns) if pd.notna(ns) else np.nan,
            "n_vslow": int(nv) if pd.notna(nv) else np.nan,
            "Sharpe": s["Sharpe Ratio"],
        })
    return pd.DataFrame(rows).set_index("Start")

params_df = extract_params_from_stats(stats)
print(params_df.tail())
params_df.to_csv("wf_params.csv")


        
