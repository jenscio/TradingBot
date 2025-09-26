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
full_df = pd.read_csv(CSV_PATH)
full_df['time'] = pd.to_datetime(full_df['time'].astype(int), unit='s', utc=True)
full_df.set_index('time', inplace=True)
full_df = full_df.tz_convert(None).sort_index()
full_df = full_df.rename(columns={
    'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'
})[['Open','High','Low','Close','Volume']].astype(float)
full_df = full_df.dropna(subset=['Open','High','Low','Close'])

# ---------- Slices ----------
TRAIN_START = "2014-12-17"
TRAIN_END   = "2020-12-31"   # inclusive
VAL_START   = "2021-01-01"
VAL_END     = "2022-12-31"   # inclusive

train_df = full_df.loc[TRAIN_START:TRAIN_END].copy()
val_df   = full_df.loc[VAL_START:VAL_END].copy()

print("Train range:", train_df.index.min(), "→", train_df.index.max(), "bars:", len(train_df))
print("Val   range:", val_df.index.min(), "→", val_df.index.max(), "bars:", len(val_df))
print(train_df.head())

# ---------- Backtest + optimiser config ----------
btng.Pool = mp.get_context('spawn').Pool

def make_bt(df):
    return Backtest(
        df,
        ARSIstrat,
        cash=1_000_000,
        commission=0.0,
        spread=0.000025,
        finalize_trades=True
    )

bt_train = make_bt(train_df)
bt_val   = make_bt(val_df)
PRINT_COLS = ['Return [%]','Sharpe Ratio','Sortino Ratio','Calmar Ratio','Max. Drawdown [%]','# Trades']

def stats_summary(stats):
    cols = [c for c in PRINT_COLS if c in stats.index]
    out = {c: float(stats.get(c, np.nan)) for c in cols}
    out['Robust Score'] = robust_score(stats)
    return out

def snap_sl(x: float) -> float:
    return float(min(SL_GRID, key=lambda g: abs(g - float(x))))

def params_key(d: dict):
    return tuple(sorted((k, int(v)) for k, v in d.items()))


def fmt_summary(s):
    def get(k, nd=2):
        v = s.get(k, np.nan)
        return f"{v:.{nd}f}" if np.isfinite(v) else "nan"
    return ("RScore {rs} | Sharpe {sh} | Sortino {so} | Calmar {ca} | "
            "DD {dd}% | Ret {ret}% | Trades {tr}").format(
        rs=get('Robust Score', 6),
        sh=get('Sharpe Ratio', 2),
        so=get('Sortino Ratio', 2),
        ca=get('Calmar Ratio', 2),
        dd=get('Max. Drawdown [%]', 2),
        ret=get('Return [%]', 2),
        tr=int(s.get('# Trades', np.nan)) if np.isfinite(s.get('# Trades', np.nan)) else 'nan'
    )

def idx_to_params(idx, names):
    vals = idx if isinstance(idx, tuple) else (idx,)
    d = dict(zip(names, vals))
    # ints for int params
    for k in ('n_fast','n_slow','n_vslow','sig_len'):
        if k in d:
            d[k] = int(d[k])
    return d


# ---------- Constraints ----------
MIN_GAP_FS = 5   # n_slow - n_fast  >= 3
MIN_GAP_SV = 7   # n_vslow - n_slow >= 7

# Constraints: enforce parameter ordering &andminimum gaps between fast/slow/vslow,
# and require positive signal length and sl
def _order_ok(p):
    return (
        (p.n_slow  - p.n_fast  >= MIN_GAP_FS) and
        (p.n_vslow - p.n_slow  >= MIN_GAP_SV) and
        (p.n_fast  < p.n_slow  < p.n_vslow)  
    )

# ---------- Search space ----------
param_space = dict(
    n_fast  = list(range(5, 26, 1)),
    n_slow  = list(range(10, 56, 1)),
    n_vslow = list(range(30, 121, 1)),
    sig_len = list(range(5, 31, 1)),
    
)

def robust_score(stats) -> float:
    """
    QQQ-robust composite objective:
      + 0.25 * tanh(Sharpe/2)
      + 0.35 * tanh(Sortino/2)
      + 0.25 * tanh(Calmar/3)
      - 0.10 * tanh(MaxDD / 20%)
      - 0.05 * tanh(LeftTail5% / 3%)
    all multiplied by min(1, #trades / 100)

    Falls back to computing Sortino from equity curve if missing.
    Requires stats._trades for left-tail penalty; if missing/empty, returns very low score.
    """

    def _get_float(key, default=np.nan):
        try:
            v = stats.get(key, default)
            if v is None or isinstance(v, (pd.Series, pd.DataFrame)):
                return default
            return float(v)
        except Exception:
            return default

    # --- Core ratios from stats (if available) ---
    sharpe = _get_float('Sharpe Ratio', np.nan)
    calmar = _get_float('Calmar Ratio', np.nan)
    max_dd_pct = abs(_get_float('Max. Drawdown [%]', np.nan))  # positive %

    # --- Sortino: use provided, else compute from equity curve ---
    sortino = _get_float('Sortino Ratio', np.nan)
    if not np.isfinite(sortino):
        # Try to compute from equity curve (downside deviation)
        try:
            ec = getattr(stats, '_equity_curve', None)
            if isinstance(ec, pd.DataFrame) and 'Equity' in ec:
                # bar-to-bar returns of equity in %
                r = ec['Equity'].pct_change().dropna()
                # downside only
                downside = r[r < 0]
                if len(downside) >= 5 and r.std(ddof=0) > 0:
                    # annualize using sqrt(N); N = bars per year ~ 252*6.5=1638 for 60-min bars US equities,
                    # but simpler: use 252 trading days for risk scale consistency.
                    N = 252.0
                    mean_r = r.mean() * N
                    downside_dev = downside.std(ddof=0) * np.sqrt(N)
                    sortino = mean_r / (downside_dev + 1e-12)
        except Exception:
            pass

    # --- Trades & left-tail loss (5th percentile of trade PnL %) ---
    trades = stats.get('_trades', None)
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return -1e9  # can’t evaluate robustness without trades

    n_trades = len(trades)

    # Trade PnL percent column (handle name variants)
    pnl_cols = [c for c in trades.columns if c.lower().replace(' ', '') in ('pnl[%]', 'pnl%', 'pnlpc')]
    if pnl_cols:
        pnl_pct = pd.to_numeric(trades[pnl_cols[0]], errors='coerce').dropna()
    else:
        pnl_pct = pd.Series([], dtype=float)

    if len(pnl_pct) >= 5:
        q05 = float(np.nanpercentile(pnl_pct.values, 5))
        left_tail = max(0.0, -q05)  # only penalize if negative
    else:
        # If we can’t estimate, be conservative but not catastrophic
        left_tail = 0.0

    # --- Helpers ---
    def squash(x, denom):
        if x is None or not np.isfinite(x):
            return 0.0
        return float(np.tanh(x / denom))

    trade_factor = min(1.0, n_trades / 100.0)

    # --- Components per spec ---
    s_sharpe   = 0.3 * squash(sharpe, 2.0)
    s_sortino  = 0.55 * squash(sortino, 2.0)
    # s_calmar   = 0.25 * squash(calmar, 3.0)
    p_dd       = 0.10 * squash((max_dd_pct or 0.0) / 100.0, 0.20)    # DD normalized to 0–1 by 100%
    p_tail     = 0.05 * squash((left_tail or 0.0) / 100.0, 0.03)     # Tail as percent

    score = trade_factor * (s_sharpe + s_sortino) - (p_dd + p_tail)

    # (Optional) light complexity penalty to de-prefer overlong signals
    try:
        sig_len = int(getattr(stats._strategy, 'sig_len', 0))
        score -= 0.0005 * max(0, sig_len - 10)
    except Exception:
        pass

    return float(score)


# ---------- Run Optimise ----------
if __name__ == '__main__':
    # Optimize on training slice
    best_stats, heatmap = bt_train.optimize(
        **param_space,
        maximize=robust_score,
        return_heatmap=True,
        constraint=_order_ok,
        method='sambo',
        max_tries=5000,
        random_state=81
    )

    print("\n=== Best on TRAIN (by robust_score) ===")
    print(best_stats[[c for c in PRINT_COLS if c in best_stats.index]])
    print(f"Robust Score: {robust_score(best_stats):.6f}")
    best = best_stats._strategy
    best_params = {
        'n_fast':  int(best.n_fast),
        'n_slow':  int(best.n_slow),
        'n_vslow': int(best.n_vslow),
        'sig_len': int(best.sig_len),
    }
    print("Best params (TRAIN):", best_params)

    # Build candidate list from heatmap
    candidates = []
    seen = set()

    if hasattr(heatmap, 'index'):
        if isinstance(heatmap, pd.Series):
            # When maximize is callable, many versions return a Series of the objective.
            # Take top M by that value directly.
            M = min(200, len(heatmap))
            top_series = heatmap.sort_values(ascending=False).head(M)
            names = heatmap.index.names
            for idx, obj in top_series.items():
                params = idx_to_params(idx, names)
                t = tuple(sorted(params.items()))
                if t in seen: 
                    continue
                seen.add(t)
                candidates.append(params)
        else:
            # DataFrame: rank by an available robust-ish proxy to preselect,
            # then re-run to compute true robust score.
            rank_cols = [c for c in ['Sortino Ratio','Calmar Ratio','Sharpe Ratio','Return [%]'] if c in heatmap.columns]
            if not rank_cols:
                # fallback: arbitrary sample
                sampled = list(heatmap.index)[:200]
            else:
                # create a simple proxy score
                hm = heatmap.copy()
                proxy = 0
                if 'Sortino Ratio' in hm.columns: proxy = proxy + (hm['Sortino Ratio'].astype(float).fillna(0))*0.5
                if 'Calmar Ratio'  in hm.columns: proxy = proxy + (hm['Calmar Ratio'].astype(float).fillna(0))*0.5
                elif 'Sharpe Ratio' in hm.columns: proxy = proxy + (hm['Sharpe Ratio'].astype(float).fillna(0))*0.3
                if 'Return [%]'    in hm.columns: proxy = proxy + (hm['Return [%]'].astype(float).fillna(0))*0.1
                hm['_proxy'] = proxy
                sampled = hm.sort_values('_proxy', ascending=False).head(min(200, len(hm))).index

            names = heatmap.index.names
            for idx in sampled:
                params = idx_to_params(idx, names)
                t = tuple(sorted(params.items()))
                if t in seen: 
                    continue
                seen.add(t)
                candidates.append(params)
    else:
        # Fallback: use best param set only
        candidates.append(best_params)

    # Re-run candidates on TRAIN and compute true robust score
    scored_train = []
    for i, params in enumerate(candidates, 1):
        st = bt_train.run(**params)
        summ = stats_summary(st)
        scored_train.append((params, summ))
    # Rank by robust score
    scored_train.sort(key=lambda x: x[1]['Robust Score'], reverse=True)
    top25_train = scored_train[:25]
    train_lookup = {}
    for rank, (params, summ) in enumerate(top25_train, 1):
        train_lookup[params_key(params)] = {
            'rank': rank,
            'rscore': summ['Robust Score']
        }

    print("\n=== TOP 25 PARAM SETS ON TRAIN (2014-12-17 → 2020-12-31) ===")
    for i, (params, summ) in enumerate(top25_train, 1):
        print(f"{i:2d}. {fmt_summary(summ)}  ->  {params}")

    # ---------- VALIDATE: Top-10 on 2021-01-01 → 2022-12-31 ----------
    top10_params = [p for p, _ in top25_train[:10]]

    print("\n=== VALIDATION of TOP 10 on 2021-01-01 → 2022-12-31 ===")
    val_results = []
    for params in top10_params:
        st_val = bt_val.run(**params)
        summ_val = stats_summary(st_val)
        # attach training rank info
        key = params_key(params)
        tr_info = train_lookup.get(key, {'rank': None, 'rscore': float('nan')})
        val_results.append((params, summ_val, tr_info))

    # Sort by validation robust score (desc)
    val_results.sort(key=lambda x: x[1]['Robust Score'], reverse=True)

    for i, (params, summ_val, tr_info) in enumerate(val_results, 1):
        tr_rank = tr_info['rank']
        tr_rs   = tr_info['rscore']
        tr_str  = f"train #{tr_rank}" if tr_rank is not None else "train n/a"
        rs_tr   = f"{tr_rs:.6f}" if np.isfinite(tr_rs) else "nan"
        print(f"{i:2d}. ({tr_str}, RScore_train {rs_tr}) {fmt_summary(summ_val)}  ->  {params}")


    # Optional: show the winner on validation explicitly
    if val_results:
        best_val_params = val_results[0][0]
        print("\nBest params on VALIDATION:", best_val_params)