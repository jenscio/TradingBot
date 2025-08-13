"""
MLP Surrogate + Monte Carlo Stress for ARSIstrat
-------------------------------------------------
- Self-contained script: includes your ARSIstrat strategy and data loading.
- Pipeline:
    1) Load hourly QQQ CSV (same path you used)
    2) Define parameter space for ARSIstrat
    3) Collect initial random train evaluations
    4) Train a tiny PyTorch MLP surrogate with L2 weight decay + early stopping
    5) Do gradient ascent on the surrogate to propose candidates
    6) Rank by a robust multi-split validation score (time splits + cost jitter)
    7) Monte Carlo block-bootstrap stress on the validation year (q25, mu, sd, p)
    8) Final untouched test on 2023

Requirements:
    pip install torch backtesting ta pandas numpy

Notes:
- Adjust N_INIT, N_ROUNDS, TOP_K, and MC_B for speed vs robustness.
- Keep bt.plot() OFF in loops (too slow). We only print stats.
"""

import os
import time
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Core libs
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# --- Strategy (from your code) ------------------------------------------------
import ta


def sma(series, length):
    s = pd.Series(series)
    return s.rolling(length).mean()


def ema(series, span):
    s = pd.Series(series)
    return s.ewm(span=span, adjust=False).mean()


def rma(series, length):
    s = pd.Series(series)
    return s.ewm(alpha=1 / length, adjust=False).mean()


def ma(series, length, ma_type: str):
    ma_type = ma_type.lower()
    if ma_type == 'sma':
        return sma(series, length)
    elif ma_type == 'ema':
        return ema(series, length)
    elif ma_type in ('rma', 'smma'):
        return rma(series, length)
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")


def augmented_rsi(src, length, smooth_len, ma_type='rma', smooth_type='ema'):
    src = pd.Series(src)

    upper = src.rolling(length, min_periods=1).max()
    lower = src.rolling(length, min_periods=1).min()
    r = upper - lower

    prev_upper = upper.shift(1)
    prev_lower = lower.shift(1)
    cond_up = upper > prev_upper
    cond_down = lower < prev_lower

    diff = src.diff()
    diff = diff.where(~cond_up & ~cond_down, other=r.where(cond_up, -r))

    num = ma(diff, length, ma_type)
    den = ma(diff.abs(), length, ma_type)

    arsi = (num / den) * 50 + 50
    signal = ma(arsi, smooth_len, smooth_type)
    return arsi, signal


def awesome_osc(high, low):
    h = pd.Series(high)
    l = pd.Series(low)
    med = (h + l) / 2.0
    return med.ewm(span=5, adjust=False).mean() - med.ewm(span=34, adjust=False).mean()


def stdev_fast_block(close):
    c = pd.Series(close)
    sma20 = c.rolling(20).mean()
    direction = np.sign(sma20.diff())
    stdev_fast = c.rolling(20).std() * direction
    stdev_signal = stdev_fast.ewm(span=9, adjust=False).mean()
    return stdev_fast, stdev_signal


def stdev2_slow_block(close):
    c = pd.Series(close)
    ma200 = c.rolling(200).mean()
    direction = np.sign(ma200.diff())
    stdev2_slow = c.rolling(200).std() * direction
    stdev2_signal = stdev2_slow.ewm(span=50, adjust=False).mean()
    return stdev2_slow, stdev2_signal


class ARSIstrat(Strategy):
    sl_pct = 0.005
    tp_pct = None
    n_fast = 9
    n_slow = 22
    n_vslow = 55
    sig_len = 14

    def init(self):
        c = self.data.Close
        h = self.data.High
        l = self.data.Low

        self.EMA_50 = self.I(ema, c, 50)
        self.TrendEMA_100 = self.I(ema, c, 100)
        self.EMA_200 = self.I(ema, c, 200)
        self.MA_200 = self.I(sma, c, 200)

        self.ARSI_fast, self.SIG_fast = self.I(
            augmented_rsi, c, self.n_fast, self.sig_len, 'rma', 'ema'
        )
        self.ARSI_slow, self.SIG_slow = self.I(
            augmented_rsi, c, self.n_slow, self.sig_len, 'rma', 'ema'
        )
        self.ARSI_vslow, self.SIG_vslow = self.I(
            augmented_rsi, c, self.n_vslow, self.sig_len, 'rma', 'ema'
        )

        self.AO = self.I(awesome_osc, h, l)
        self.Stdev_fast, self.Stdev_signal = self.I(stdev_fast_block, c)
        self.Stdev2_slow, self.Stdev2_signal = self.I(stdev2_slow_block, c)

    def next(self):
        if len(self.data) < 205:
            return

        main_long = (self.data.Close[-1] > self.MA_200[-1]) and (self.MA_200[-1] > self.MA_200[-4])
        ursi3 = crossover(self.ARSI_fast, self.ARSI_slow)
        ursi4 = crossover(self.ARSI_vslow, self.SIG_vslow)
        stdev_l = crossover(self.Stdev_fast, self.Stdev_signal)
        stdev2_l = crossover(self.Stdev2_slow, self.Stdev2_signal)

        def xover_scalar(arr, lvl):
            a = np.asarray(arr)
            return a[-1] > lvl and a[-2] <= lvl

        def xunder_scalar(arr, lvl):
            a = np.asarray(arr)
            return a[-1] < lvl and a[-2] >= lvl

        ao_long = xover_scalar(self.AO, 0.0)
        ao_short = xunder_scalar(self.AO, 0.0)

        ema50_s = crossover(self.EMA_50, self.data.Close)
        ema200_s = crossover(self.EMA_200, self.data.Close)

        long_raw = main_long or ursi3 or ursi4 or ao_long or stdev_l or stdev2_l
        short_raw = crossover(self.SIG_vslow, self.ARSI_vslow) or ao_short or ema50_s or ema200_s

        anchor = self.data.Close[-1]

        if not self.position:
            if long_raw:
                sl = anchor * (1 - self.sl_pct)
                self.buy(sl=sl)
            elif short_raw:
                sl = anchor * (1 + self.sl_pct)
                self.sell(sl=sl)
        else:
            if self.position.is_long and short_raw:
                sl = anchor * (1 + self.sl_pct)
                self.position.close(); self.sell(sl=sl)
            elif self.position.is_short and long_raw:
                sl = anchor * (1 - self.sl_pct)
                self.position.close(); self.buy(sl=sl)


# --- Data loading -------------------------------------------------------------

def load_data(csv_path: str, start='2018-01-01', end_excl='2024-01-01'):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index < end_excl)].copy()
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    return df


# --- Backtest helpers ---------------------------------------------------------

def make_strat(p):
    class Tuned(ARSIstrat):
        sl_pct = float(p['sl_pct'])
        n_fast = int(p['n_fast'])
        n_slow = int(p['n_slow'])
        n_vslow = int(p['n_vslow'])
        sig_len = int(p['sig_len'])
    return Tuned


def run_bt(data: pd.DataFrame, params: dict, spread: float = 0.0001):
    Strat = make_strat(params)
    bt = Backtest(data, Strat, cash=1_000_000, commission=0.0, spread=spread, finalize_trades=True)
    stats = bt.run()
    sharpe = float(stats.get('Sharpe Ratio', 0.0))
    # turnover proxy
    trades = int(getattr(stats, '_trades', pd.DataFrame()).shape[0]) if hasattr(stats, '_trades') else 0
    days = max(1, (data.index[-1] - data.index[0]).days)
    tpd = trades / days
    score = sharpe
    return score, tpd, stats


# --- Parameter space ----------------------------------------------------------
SPACE = {
    'sl_pct': (0.001, 0.02, 'float'),  # 0.1% .. 2%
    'n_fast': (5, 25, 'int'),
    'n_slow': (10, 55, 'int'),
    'n_vslow': (30, 120, 'int'),
    'sig_len': (5, 30, 'int'),
}
ORDER = list(SPACE.keys())


def encode(params_dict):
    z = []
    for k in ORDER:
        lo, hi, typ = SPACE[k]
        v = float(params_dict[k])
        v = (v - lo) / (hi - lo)
        z.append(np.clip(v, 0, 1))
    return np.array(z, dtype=np.float32)


def decode(z):
    out = {}
    for i, k in enumerate(ORDER):
        lo, hi, typ = SPACE[k]
        v = lo + float(np.clip(z[i], 0, 1)) * (hi - lo)
        if typ == 'int':
            v = int(round(v))
            v = int(min(max(v, int(lo)), int(hi)))
        else:
            v = float(v)
        out[k] = v
    return out


def sample_params(n):
    Ps = []
    for _ in range(n):
        d = {}
        for k, (lo, hi, typ) in SPACE.items():
            if typ == 'int':
                d[k] = int(np.random.randint(int(lo), int(hi) + 1))
            else:
                d[k] = float(np.random.uniform(lo, hi))
        Ps.append(d)
    return Ps


# --- Surrogate (PyTorch) ------------------------------------------------------
try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise SystemExit("PyTorch not installed. Run: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu")


class Surrogate(nn.Module):
    def __init__(self, d, p_dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_surrogate(X, y, steps=2000, lr=1e-3, wd=1e-3, val_frac=0.2, patience=200):
    from copy import deepcopy
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)
    n = len(X)
    if n < 10:
        raise ValueError("Need at least 10 samples to fit surrogate")
    idx = np.random.permutation(n)
    k = int(n * (1 - val_frac))
    tr, va = idx[:k], idx[k:]
    Xt = torch.tensor(X[tr]); yt = torch.tensor(y[tr])
    Xv = torch.tensor(X[va]); yv = torch.tensor(y[va])

    model = Surrogate(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss = nn.MSELoss()

    best = (1e9, None)
    bad = 0
    for _ in range(steps):
        model.train(); opt.zero_grad()
        pred = model(Xt); l = loss(pred, yt); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss(model(Xv), yv).item()
            if v < best[0]:
                best = (v, deepcopy(model.state_dict())); bad = 0
            else:
                bad += 1
            if bad > patience:
                break
    model.load_state_dict(best[1])
    model.eval()
    return model


def argmax_on_surrogate(model, starts, iters=400, lr=0.05):
    best_z, best_pred = None, -1e9
    for z0 in starts:
        z = torch.tensor(z0.copy(), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            z_c = torch.clamp(z, 0.0, 1.0)
            pred = model(z_c)
            loss = -pred
            loss.backward(); opt.step()
        with torch.no_grad():
            z_c = torch.clamp(z, 0.0, 1.0)
            pred = model(z_c).item()
            if pred > best_pred:
                best_pred, best_z = pred, z_c.detach().numpy()
    return best_z, best_pred


# --- Robust multi-split validation -------------------------------------------
SPLITS = [
    ('2018-01-01', '2020-12-31', '2021-01-01', '2021-06-30'),
    ('2018-07-01', '2021-06-30', '2021-07-01', '2021-12-31'),
    ('2019-01-01', '2021-12-31', '2022-01-01', '2022-06-30'),
    ('2019-07-01', '2022-06-30', '2022-07-01', '2022-12-31'),
]


def slice_df(df, s, e):
    return df.loc[s:e]


def robust_validation_score(df_full, p):
    sharpes, tpds = [] , []
    for (tr_s, tr_e, val_s, val_e) in SPLITS:
        train = slice_df(df_full, tr_s, tr_e)
        if len(train) > 0:
            train = train.iloc[:-5]  # embargo 5 bars
        valid = slice_df(df_full, val_s, val_e)
        # friction jitter
        for sp in [0.00008, 0.0001, 0.00012]:
            s, t, _ = run_bt(valid, p, spread=sp)
            sharpes.append(s)
            tpds.append(t)
    sharpes = np.array(sharpes, float)
    tpds = np.array(tpds, float)
    q25 = float(np.quantile(sharpes, 0.25))
    std = float(np.std(sharpes))
    tpd = float(np.median(tpds))
    return q25 - 0.5 * std - 0.02 * tpd


# --- Monte Carlo: block bootstrap --------------------------------------------

def block_bootstrap(df, block_len=48, out_len=None, seed=None):
    rng = np.random.default_rng(seed)
    N = len(df)
    if out_len is None:
        out_len = N
    max_start = max(1, N - block_len)
    starts = rng.integers(0, max_start, size=(out_len // block_len + 2))
    pieces = [df.iloc[s:s + block_len] for s in starts]
    out = pd.concat(pieces, axis=0).iloc[:out_len].copy()
    freq = pd.infer_freq(df.index) or 'H'
    out.index = pd.date_range(start=df.index[0], periods=len(out), freq=freq)
    return out


def mc_score(df, params, B=200, block_len=48, spread_base=0.0001):
    sharpes = []
    for b in range(B):
        spread = spread_base * (0.8 + 0.4 * np.random.rand())
        boot = block_bootstrap(df, block_len=block_len, seed=b)
        s, _, _ = run_bt(boot, params, spread=spread)
        sharpes.append(s)
    sharpes = np.array(sharpes, float)
    mu = float(np.mean(sharpes))
    sd = float(np.std(sharpes))
    q25 = float(np.quantile(sharpes, 0.25))
    p = (1 + np.sum(sharpes <= 0.0)) / (len(sharpes) + 1)
    return {"mu": mu, "sd": sd, "q25": q25, "p": float(p)}


# --- Main pipeline ------------------------------------------------------------

def main():
    np.random.seed(42)

    # 0) Load data (adjust path if needed)
    csv_path = 'BATS_QQQ, 60_a45be.csv'
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    full_df = load_data(csv_path, start='2018-01-01', end_excl='2024-01-01')

    # Define periods
    df_train = full_df.loc['2018-01-01':'2021-12-31']
    df_valid_full = full_df.loc['2022-01-01':'2022-12-31']  # for MC
    df_test = full_df.loc['2023-01-01':'2023-12-31']

    # 1) Initial random design
    N_INIT = 60
    pool = sample_params(N_INIT)
    X, y = [], []
    t0 = time.perf_counter()
    for p in pool:
        s_train, _, _ = run_bt(df_train, p)
        X.append(encode(p)); y.append(s_train)
    print(f"Initial evals: {len(y)} done in {time.perf_counter()-t0:.1f}s. Best train={max(y):.3f}")

    # 2) Surrogate refinement rounds
    N_ROUNDS = 4
    TOP_K = 8
    N_STARTS = 16
    for r in range(N_ROUNDS):
        surr = fit_surrogate(X, y, steps=1500, lr=1e-3, wd=1e-3, val_frac=0.2, patience=150)
        starts = [np.random.rand(len(ORDER)) for _ in range(N_STARTS - 4)]
        # seed from best-so-far
        top_idx = np.argsort(y)[-4:]
        starts += [X[i] for i in top_idx]
        cand = []
        for z0 in starts:
            z_star, _ = argmax_on_surrogate(surr, [z0], iters=250, lr=0.05)
            cand.append(z_star)
        # dedupe
        uniq = []
        for z in cand:
            if all(np.linalg.norm(z - u) > 0.05 for u in uniq):
                uniq.append(z)
        preds = surr(torch.tensor(np.array(uniq), dtype=torch.float32)).detach().numpy()
        order = np.argsort(preds)[::-1][:TOP_K]
        added = 0
        for idx in order:
            p = decode(uniq[idx])
            s_train, _, _ = run_bt(df_train, p)
            X.append(encode(p)); y.append(s_train)
            added += 1
        print(f"Round {r+1}: added {added}, best train={max(y):.3f}")

    # 3) Candidate set -> robust multi-split validation
    candidates = [decode(x) for x in X]
    val_scores = []
    for p in candidates:
        vs = robust_validation_score(full_df, p)
        val_scores.append(vs)
    idxs = np.argsort(val_scores)[::-1]
    K_FINAL = 5
    finalists = [candidates[i] for i in idxs[:K_FINAL]]
    print("Top finalists by robust validation:")
    for i, p in enumerate(finalists, 1):
        print(f"{i}. {p} | robust_score={val_scores[idxs[i-1]]:.3f}")

    # 4) Monte Carlo on validation year
    MC_B = 200
    BLOCK_LEN = 48  # 2 trading days for hourly bars
    mc_results = []
    for p in finalists:
        res = mc_score(df_valid_full, p, B=MC_B, block_len=BLOCK_LEN, spread_base=0.0001)
        mc_results.append((res, p))
        print(f"MC for {p}: q25={res['q25']:.3f} mu={res['mu']:.3f} sd={res['sd']:.3f} p={res['p']:.3f}")

    # Select by conservative metric (q25 first, then mu - 0.5*sd)
    mc_results.sort(key=lambda t: (t[0]['q25'], t[0]['mu'] - 0.5 * t[0]['sd']), reverse=True)
    best_res, best_p = mc_results[0]
    print("\n=== Selected after Monte Carlo ===")
    print(best_p)
    print(best_res)
    print(f"Winner p-value: {best_res['p']:.4f}")


    # 5) Final untouched test on 2023
    test_score, _, test_stats = run_bt(df_test, best_p)
    print("\n=== Final Test (2023) ===")
    print(f"Test score (Sharpe-penalty): {test_score:.3f}")
    print(test_stats)


if __name__ == '__main__':
    main()
