import os
import time
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
import yfinance as yf

# Core libs
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from tradingbot.strategies.strategy import ARSIstrat
# --- Data loading -------------------------------------------------------------

def load_data(csv_path: str, start='2018-01-01', end_excl='2024-01-01'):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index < end_excl)].copy()
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


# --- Backtest helpers ---------------------------------------------------------

def make_strat(p):
    p = _enforce_order_lt(p)
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

# --- Parameter constraints beyond ordering ---
MIN_GAP_FS = 3   # slow - fast  >= 3
MIN_GAP_SV = 7   # vslow - slow >= 7

# MC-Dropout uncertainty
UCB_BETA = 0.5          # explore weight; try 0.3–1.0
MC_SAMPLES = 64         # dropout passes

def predict_mu_sigma(model, Z, n=MC_SAMPLES):
    model.train()  # enable dropout at inference
    preds = []
    with torch.no_grad():
        Xz = torch.tensor(Z, dtype=torch.float32)
        for _ in range(n):
            preds.append(model(Xz).cpu().numpy())
    P = np.stack(preds, 0)   # [n_mc, n_points]
    return P.mean(0), P.std(0)
# --- add near your SPACE/ORDER section ---------------------------------------

def _enforce_order_lt(p):
    """
    Enforce n_fast < n_slow < n_vslow with minimum gaps:
    n_slow - n_fast >= MIN_GAP_FS, n_vslow - n_slow >= MIN_GAP_SV
    while staying inside SPACE bounds.
    """
    f_lo, f_hi, _ = SPACE['n_fast']
    s_lo, s_hi, _ = SPACE['n_slow']
    v_lo, v_hi, _ = SPACE['n_vslow']

    nf = int(np.clip(int(round(p['n_fast'])), f_lo, f_hi))

    ns_min = max(s_lo, nf + MIN_GAP_FS)
    ns = int(np.clip(int(round(p['n_slow'])), ns_min, s_hi))
    # tighten nf after ns clamp
    nf = min(nf, ns - MIN_GAP_FS)

    nv_min = max(v_lo, ns + MIN_GAP_SV)
    nv = int(np.clip(int(round(p['n_vslow'])), nv_min, v_hi))
    # tighten ns after nv clamp
    ns = min(ns, nv - MIN_GAP_SV)

    out = dict(p)
    out['n_fast'], out['n_slow'], out['n_vslow'] = nf, ns, nv
    return out



def encode(params_dict):
    z = []
    for k in ORDER:
        lo, hi, typ = SPACE[k]
        v = float(params_dict[k])
        v = (v - lo) / (hi - lo)
        z.append(np.clip(v, 0, 1))
    return np.array(z, dtype=np.float32)


# --- replace your decode() with this -----------------------------------------

def decode(z):
    out = {}
    for i, k in enumerate(ORDER):
        lo, hi, typ = SPACE[k]
        v = lo + float(np.clip(z[i], 0, 1)) * (hi - lo)
        if typ == 'int':
            v = int(min(max(int(round(v)), int(lo)), int(hi)))
        else:
            v = float(v)
        out[k] = v
    return _enforce_order_lt(out)



# --- replace your sample_params() with this ----------------------------------

def sample_params(n):
    Ps = []
    f_lo, f_hi, _ = SPACE['n_fast']
    s_lo, s_hi, _ = SPACE['n_slow']
    v_lo, v_hi, _ = SPACE['n_vslow']

    # ensure feasibility when sampling
    nf_max = min(
        f_hi,
        s_hi - MIN_GAP_FS,
        v_hi - (MIN_GAP_FS + MIN_GAP_SV)
    )
    for _ in range(n):
        nf = int(np.random.randint(f_lo, nf_max + 1))

        ns_min = max(s_lo, nf + MIN_GAP_FS)
        ns_max = min(s_hi, v_hi - MIN_GAP_SV)
        ns = int(np.random.randint(ns_min, ns_max + 1))

        nv_min = max(v_lo, ns + MIN_GAP_SV)
        nv = int(np.random.randint(nv_min, v_hi + 1))

        d = {
            'sl_pct': float(np.random.uniform(*SPACE['sl_pct'][:2])),
            'n_fast': nf,
            'n_slow': ns,
            'n_vslow': nv,
            'sig_len': int(np.random.randint(int(SPACE['sig_len'][0]), int(SPACE['sig_len'][1]) + 1)),
        }
        Ps.append(_enforce_order_lt(d))
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

TRAIN_CV_FOLDS = [
    ('2018-01-01','2019-06-30'),
    ('2019-07-01','2020-12-31'),
    ('2020-01-01','2021-06-30'),
    ('2020-07-01','2021-12-31'),
]

def robust_on_train(df_train, p, spreads=(0.00008, 0.00010, 0.00012)):
    sharpes, tpds = [], []
    for (s, e) in TRAIN_CV_FOLDS:
        valid = df_train.loc[s:e]
        if valid.empty:
            continue
        for sp in spreads:
            s_, t_, _ = run_bt(valid, p, spread=sp)
            sharpes.append(float(s_)); tpds.append(float(t_))
    sharpes = np.array(sharpes, float); tpds = np.array(tpds, float)
    if sharpes.size == 0:
        return -1e9  # degenerate safety
    q25 = float(np.quantile(sharpes, 0.25))
    std = float(np.std(sharpes))
    tpd = float(np.median(tpds))
    return q25 - 0.5*std - 0.02*tpd

# --- Visualization of Top-25 vs Buy&Hold ------------------------------------

import matplotlib.pyplot as plt
import matplotlib as mpl

def buy_hold_price(df_slice: pd.DataFrame, label: str = 'QQQ Buy&Hold') -> pd.Series:
    s = df_slice.loc[:, 'Close']
    # If duplicate 'Close' columns exist, df_slice['Close'] returns a DataFrame — take the first.
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors='coerce')
    s.name = label
    return s

def equity_curve_raw(df_slice: pd.DataFrame, p: dict, label: str) -> pd.Series:
    # raw equity curve from Backtesting.py
    _, _, stats = run_bt(df_slice, p)
    eq = stats._equity_curve['Equity'].astype(float).copy()
    eq.name = label
    return eq

def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # If yfinance returned MultiIndex columns, drop the non-OHLCV level
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        fields = {'Open','High','Low','Close','Volume'}

        if fields.issubset(lvl0):
            df = df.droplevel(1, axis=1)  # keep level 0 = fields
        elif fields.issubset(lvl1):
            df = df.droplevel(0, axis=1)  # keep level 1 = fields
        else:
            raise ValueError("Can't find OHLCV fields in MultiIndex columns")

    # Drop Adj Close if present; Backtesting wants exactly O/H/L/C/V
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])
    # Ensure ordering and types
    cols = ['Open','High','Low','Close','Volume']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after flatten: {missing}")
    df = df[cols].astype(float)
    return df

def plot_top25_fair(full_df, params_list, start, end_excl, title, outfile, bh_label='QQQ Buy&Hold'):
    df_slice = full_df.loc[(full_df.index >= start) & (full_df.index < end_excl)].copy()
    if df_slice.empty:
        raise ValueError(f"No data in [{start}, {end_excl})")

    # tz + columns hygiene
    if getattr(df_slice.index, "tz", None) is not None:
        df_slice = df_slice.tz_convert("Europe/Zurich").tz_localize(None)
    df_slice = ensure_ohlcv(df_slice).dropna(subset=["Open","High","Low","Close","Volume"])

    # raw series
    bh_px = buy_hold_price(df_slice, label=bh_label)
    labels, strat_eqs = [], []
    for k, p in enumerate(params_list, 1):
        lab = f"P{k:02d} f{p['n_fast']}-s{p['n_slow']}-v{p['n_vslow']}"
        labels.append(lab)
        strat_eqs.append(equity_curve_raw(df_slice, p, lab))

    # ---- HARD ALIGNMENT: start everyone at the same timestamp ----
    # pick the latest first-valid timestamp among B&H and all strategies
    t0_candidates = [bh_px.first_valid_index()] + [s.first_valid_index() for s in strat_eqs]
    t0_anchor = max(t for t in t0_candidates if t is not None)

    # build common index from that anchor onward
    common_idx = bh_px.loc[t0_anchor:].index
    for s in strat_eqs:
        common_idx = common_idx.intersection(s.loc[t0_anchor:].index)
    if len(common_idx) == 0:
        raise ValueError("Empty common index after alignment. Check data ranges.")
    common_idx = common_idx.sort_values()
    t0 = common_idx[0]

    # rebase all to 1.0 at t0
    curves = []
    bh = (bh_px.loc[common_idx] / bh_px.loc[t0]); bh.name = bh_label
    curves.append(bh)
    for lab, s in zip(labels, strat_eqs):
        s = (s.loc[common_idx] / s.loc[t0]); s.name = lab
        curves.append(s)

    df_plot = pd.concat(curves, axis=1)
    anchor = df_plot.index[0]
    print("Anchor:", anchor)
    bh0 = bh_px.first_valid_index()
    delay_days = (anchor - bh0).total_seconds() / 86400
    print(f"Buy&Hold started {delay_days:.2f} calendar days before anchor.")

        # 100% remove any residual offset (both visually and numerically)
    # 1) de-dup index just in case
    df_plot = df_plot.loc[~df_plot.index.duplicated(keep='first')]

    # 2) rebase EVERYTHING to exactly 1.0 at the first timestamp
    df_plot = df_plot / df_plot.iloc[0]

    # 3) optional: force x-axis to start at that anchor
    plt.xlim(df_plot.index[0], df_plot.index[-1])

    # plot (unchanged)
    base_colors = mpl.colormaps['tab20'].colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(params_list))]
    plt.figure(figsize=(13, 7))
    for i, lab in enumerate(labels):
        plt.plot(df_plot.index, df_plot[lab], label=lab, linewidth=1.0, alpha=0.9, color=colors[i])
    plt.plot(df_plot.index, df_plot[bh_label], label=bh_label, linewidth=2.2, color='black')
    plt.title(title)
    plt.ylabel("Growth of $1 (rebased at common start)")
    plt.xlabel("Time")
    plt.legend(ncol=2, fontsize=8, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()


def plot_top25_cac40_with_delayed_bh(cac_df, params_list, delay_days=15,
                                     title="CAC40 — Top 25 vs Buy&Hold (+15d delay)",
                                     outfile="top25_cac40_delay.png",
                                     bh_label="CAC40 Buy&Hold"):
    # hygiene
    if getattr(cac_df.index, "tz", None) is not None:
        cac_df = cac_df.tz_convert("Europe/Zurich").tz_localize(None)
    df = ensure_ohlcv(cac_df).dropna(subset=["Open","High","Low","Close","Volume"])

    # strategies
    labels, eqs = [], []
    for k, p in enumerate(params_list, 1):
        lab = f"P{k:02d} f{p['n_fast']}-s{p['n_slow']}-v{p['n_vslow']}"
        labels.append(lab)
        eqs.append(equity_curve_raw(df, p, lab))

    # common start across strategies
    s0s = [s.first_valid_index() for s in eqs if s is not None]
    if not s0s:
        raise ValueError("No valid strategy equity curves.")
    t_strat = max(s0s)  # latest strategy warm-up completion

    # intersection of strategy indexes from t_strat onward
    common_idx = None
    for s in eqs:
        si = s.loc[t_strat:].index
        common_idx = si if common_idx is None else common_idx.intersection(si)
    if common_idx is None or len(common_idx) == 0:
        raise ValueError("Empty common index across strategies.")
    common_idx = common_idx.sort_values()

    # rebase strategies at t_strat
    curves = []
    for lab, s in zip(labels, eqs):
        s = s.loc[common_idx]
        s = s / s.iloc[0]
        s.name = lab
        curves.append(s)

    # CAC40 Buy&Hold that starts 15 days AFTER t_strat
    bh_px = buy_hold_price(df, label=bh_label)
    t_bh = pd.Timestamp(t_strat) + pd.Timedelta(days=delay_days)
    bh_slice = bh_px.loc[t_bh:]
    if bh_slice.empty:
        raise ValueError(f"No CAC40 data at/after {t_bh}.")
    bh = bh_slice / bh_slice.iloc[0]
    bh.name = bh_label
    curves.append(bh)

    # combine & plot
    df_plot = pd.concat(curves, axis=1)
    df_plot = df_plot.loc[~df_plot.index.duplicated(keep='first')]

    base_colors = mpl.colormaps['tab20'].colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    plt.figure(figsize=(13, 7))
    for i, lab in enumerate(labels):
        plt.plot(df_plot.index, df_plot[lab], label=lab, linewidth=1.0, alpha=0.9, color=colors[i])
    if bh_label in df_plot:
        plt.plot(df_plot.index, df_plot[bh_label], label=bh_label, linewidth=2.2, color='black')

    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Time")
    plt.legend(ncol=2, fontsize=8, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.show()





def main():
    np.random.seed(42)

    # 0) Load data (adjust path if needed)
    csv_path = 'data/BATS_QQQ, 60_a45be.csv'
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    full_df = load_data(csv_path, start='2018-01-01', end_excl='2024-07-15')

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
        s_train = robust_on_train(df_train, p)  # <-- changed
        X.append(encode(p)); y.append(s_train)
    print(f"Initial evals: {len(y)} done in {time.perf_counter()-t0:.1f}s. Best train-robust={max(y):.3f}")
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
        # dedupe -> uniq (unchanged)

        Z = np.array(uniq, dtype=np.float32)
        mu, sig = predict_mu_sigma(surr, Z, n=MC_SAMPLES)
        ucb = mu + UCB_BETA * sig
        order = np.argsort(ucb)[::-1][:TOP_K]
        added = 0
        for idx in order:
            p = decode(Z[idx])
            s_train = robust_on_train(df_train, p)  # robust label
            X.append(encode(p)); y.append(s_train)
            added += 1
        print(f"Round {r+1}: added {added}, best train-robust={max(y):.3f}")


    # 3) Candidate set -> robust multi-split validation
    candidates = [decode(x) for x in X]
    val_scores = []
    for p in candidates:
        vs = robust_validation_score(full_df, p)
        val_scores.append(vs)
    idxs = np.argsort(val_scores)[::-1]
    K_FINAL = 25
    top_idxs = np.argsort(val_scores)[::-1][:K_FINAL]

    print("\nTop 25 by robust validation (with 2023 test Sharpe):")
    df_test = full_df.loc['2023-01-01':'2023-12-31']  # ensure defined here
    top_rows = []
    for rank, i in enumerate(top_idxs, 1):
        p = candidates[i]
        sharpe_2023, _, _ = run_bt(df_test, p)   # run_bt returns Sharpe
        print(f"{rank:2d}. {p} | robust={val_scores[i]:.3f} | Sharpe(2023)={sharpe_2023:.3f}")
        top_rows.append((rank, p, float(val_scores[i]), float(sharpe_2023)))

    best_p = candidates[top_idxs[0]]  # define before final test

    # final test printout (unchanged semantics)
    test_score, _, test_stats = run_bt(df_test, best_p)
    print("\n=== Final Test (2023) ===")
    print(f"Sharpe: {test_score:.3f}")
    print(test_stats)

    top_params = [candidates[i] for i in top_idxs]

    plot_top25_fair(full_df, top_params, '2018-01-01','2024-01-01',
                'QQQ: Growth of $1 — Top-25 Params vs Buy&Hold (2018–2023)',
                'top25_2018_2023.png', bh_label='QQQ Buy&Hold')

    plot_top25_fair(full_df, top_params, '2024-01-01','2025-07-07',
                    'QQQ: Growth of $1 — Top-25 Params vs Buy&Hold (2024–present)',
                    'top25_2024_2025.png', bh_label='QQQ Buy&Hold')
    # smi_df = yf.download("^SSMI", period="730d", interval="1h", auto_adjust=False, progress=False)
    ticker = "URTH"          # or "SWDA.L", "EUNL.DE", "SWDA.SW"
    msci_df = yf.download("^FCHI", period="730d", interval="1h",
                     auto_adjust=False, progress=False)

    # Flatten if needed
    msci_df = msci_df[["Open","High","Low","Close","Volume"]]

    # SMI plot
    plot_top25_fair(msci_df, top_params, str(msci_df.index.min()),
                    str(msci_df.index.max() + pd.Timedelta(hours=1)),
                    'CAC40 — Top 25 vs Buy&Hold (last ~2y hourly)',
                    'top25_msci_hourly.png', bh_label='CAC40 Buy&Hold')

    cac_df = yf.download("^FCHI", period="730d", interval="1h",
                     auto_adjust=False, progress=False)

    plot_top25_cac40_with_delayed_bh(
        cac_df, top_params, delay_days=30,
        title="CAC40 — Top 25 vs Buy&Hold (BH starts +15d)",
        outfile="top25_cac40_delay.png",
        bh_label="CAC40 Buy&Hold"
    )
    plot_top25_cac40_with_delayed_bh(
        cac_df, top_params, delay_days=45,
        title="CAC40 — Top 25 vs Buy&Hold (BH starts +15d)",
        outfile="top25_cac40_delay.png",
        bh_label="CAC40 Buy&Hold"
    )

if __name__ == '__main__':
     main()