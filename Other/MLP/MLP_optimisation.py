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
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from Strategy.strategy import ARSIstrat
# --- Data loading -------------------------------------------------------------
CUTOFF = "2022-01-01"

ASSETS = {
    # "SP": ("csv", os.path.join("CSV_files", "SP_SPX, 60_47660.csv"),
       #      {"start": "2014-01-02", "end_excl": CUTOFF}),
    "QQQ": ("csv", os.path.join("CSV_files", "BATS_QQQ, 60_a45be.csv"),
             {"start": "2014-12-17", "end_excl": CUTOFF}),
}

def load_data(csv_path: str, start='2014-12-17', end_excl='2022-01-01'):
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

# ==== DATA LOADERS ============================================================
def load_asset(name, kind, src, kwargs):
    if kind == "csv":
        df = load_data(src, **kwargs)
    elif kind == "yf":
        df = yf.download(src, auto_adjust=False, progress=False, **kwargs)
        if getattr(df.index, "tz", None) is not None:
            df = df.tz_convert("Europe/Zurich").tz_localize(None)
        df = ensure_ohlcv(df)
    else:
        raise ValueError(f"Unknown kind {kind}")
    # Backtesting wants Volume present; Yahoo indices often have NaN/0 → fill
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    return df

def load_all_assets():
    out = {}
    for name, (kind, src, kw) in ASSETS.items():
        df = load_asset(name, kind, src, kw)
        # light cleanup
        df = df.dropna(subset=["Open","High","Low","Close"]).copy()
        out[name] = df
    return out

# ==== ROBUST SCORING (per-asset CV, then aggregate across assets) ============
def _quantile_folds(df_idx, k=4):
    # k roughly-equal contiguous time folds
    qs = np.linspace(0, 1, k+1)
    cuts = [df_idx[int(q*(len(df_idx)-1))] for q in qs]
    folds = [(cuts[i], cuts[i+1]) for i in range(k)]
    return folds

def robust_cv_score_on_df(df, p, spreads=(0.00008,0.00010,0.00012), k=4):
    if len(df) < 200:
        return -1e9
    folds = _quantile_folds(df.index, k=k)
    sharpes, tpds = [], []
    for (s, e) in folds:
        valid = df.loc[s:e]
        if valid.empty:
            continue
        # embargo a few bars
        if len(valid) > 5:
            valid = valid.iloc[5:]
        for sp in spreads:
            s_, t_, _ = run_bt(valid, p, spread=sp)
            sharpes.append(float(s_)); tpds.append(float(t_))
    if not sharpes:
        return -1e9
    sharpes = np.array(sharpes); tpds = np.array(tpds)
    q25 = float(np.quantile(sharpes, 0.25))
    std = float(np.std(sharpes))
    tpd = float(np.median(tpds))
    return q25 - 0.5*std - 0.02*tpd

def robust_score_across_assets(assets_dict, p):
    # score each asset, then aggregate robustly
    scores = []
    for name, df in assets_dict.items():
        s = robust_cv_score_on_df(df, p)
        scores.append(s)
    scores = np.array(scores, float)
    if (scores <= -1e8).all():
        return -1e9
    # median for robustness; small penalty for dispersion across assets
    return float(np.median(scores) - 0.15*np.std(scores))



# --- Parameter space ----------------------------------------------------------
SPACE = {
    'sl_pct': (0.00, 0.02, 'float'),  # 0.1% .. 2%
    'n_fast': (3, 50, 'int'),
    'n_slow': (10, 100, 'int'),
    'n_vslow': (30, 200, 'int'),
    'sig_len': (5, 50, 'int'),
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
import numpy as np
import torch
import torch.nn as nn

# ---------------- Model ----------------
class Surrogate(nn.Module):
    def __init__(self, d, width=256, depth=8, p_dropout=0.10):
        super().__init__()
        self.inp = nn.Linear(d, width)
        blocks = []
        for i in range(depth):
            blocks += [
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.Dropout(p_dropout),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Linear(d, width) if d != width else nn.Identity()
        self.out = nn.Linear(width, 1)

    def forward(self, x):
        h0 = self.inp(x)
        h = h0
        # residual every two layers (after Linear)
        for i in range(0, len(self.blocks), 4):
            block = self.blocks[i:i+4]
            h = block(h)  # LN -> GELU -> Linear -> Dropout
            if i % 8 == 0:   # every second block, add residual (tune as you like)
                h = h + h0
        return self.out(h).squeeze(-1)


# ---------------- Fit function ----------------
def fit_surrogate(
    X, y, 
    steps=2500, lr=3e-4, wd=1e-3, val_frac=0.2, patience=250, clip=1.0
):
    """
    Fit surrogate with Surrogate-B architecture but:
      - standardize X and y
      - cosine LR schedule
      - gradient clipping
      - AdamW optimizer
      - patience-based early stopping
    """
    from copy import deepcopy
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)

    # --- normalize inputs and outputs ---
    Xm, Xs = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    ym, ys = y.mean(), y.std() + 1e-8
    Xn, yn = (X - Xm) / Xs, (y - ym) / ys

    n = len(Xn)
    if n < 10:
        raise ValueError("Need at least 10 samples to fit surrogate")
    idx = np.random.permutation(n)
    k = int(n * (1 - val_frac))
    tr, va = idx[:k], idx[k:]

    Xt = torch.tensor(Xn[tr]); yt = torch.tensor(yn[tr])
    Xv = torch.tensor(Xn[va]); yv = torch.tensor(yn[va])

    model = Surrogate(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss = nn.MSELoss()

    best = (1e9, None)
    bad = 0
    for _ in range(steps):
        # train step
        model.train(); opt.zero_grad(set_to_none=True)
        pred = model(Xt); l = loss(pred, yt)
        l.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step(); sched.step()

        # validation
        model.eval()
        with torch.no_grad():
            v = loss(model(Xv), yv).item()
            if v < best[0]:
                best = (v, deepcopy(model.state_dict())); bad = 0
            else:
                bad += 1
            if bad > patience:
                break

    # restore best weights
    model.load_state_dict(best[1])
    model.eval()

    # attach scalers
    model._scalers = (torch.tensor(Xm), torch.tensor(Xs), float(ym), float(ys))
    return model


# ---------------- Prediction helpers ----------------
def predict_mu_sigma(model, Z, n=MC_SAMPLES):
    model.train()
    Xm, Xs, ym, ys = model._scalers
    Xz = (torch.tensor(Z, dtype=torch.float32) - Xm) / Xs
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(model(Xz).cpu().numpy())
    P = np.stack(preds, 0) * ys + ym
    return P.mean(0), P.std(0)

def _model_predict_raw(model, Z):
    """
    Predict de-normalized surrogate outputs for given encoded params Z.
    Z: np.ndarray of shape [n_points, d] in [0,1] scale.
    """
    Xm, Xs, ym, ys = model._scalers
    Xz = (torch.tensor(Z, dtype=torch.float32) - Xm) / Xs
    with torch.no_grad():
        pred_n = model(Xz).cpu().numpy()
    return pred_n * ys + ym

def argmax_on_surrogate(model, starts, iters=400, lr=0.05):
    best_z, best_pred = None, -1e9
    Xm, Xs, ym, ys = model._scalers
    for z0 in starts:
        z = torch.tensor(z0.copy(), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            z_c = torch.clamp(z, 0.0, 1.0)
            z_in = (z_c - Xm) / Xs        # normalize input
            pred_norm = model(z_in)       # prediction in normalized space
            loss = -pred_norm
            loss.backward()
            opt.step()
        with torch.no_grad():
            z_c = torch.clamp(z, 0.0, 1.0)
            pred = _model_predict_raw(model, z_c[None, :])[0]  # denormalized
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





def main():
    np.random.seed(42)

    # Load all assets
    assets = load_all_assets()
    print({k: (assets[k].index.min(), assets[k].index.max(), len(assets[k])) for k in assets})

    # Initial random design
    N_INIT = 60
    pool = sample_params(N_INIT)
    X, y = [], []
    t0 = time.perf_counter()
    for p in pool:
        s_train = robust_score_across_assets(assets, p)
        X.append(encode(p)); y.append(s_train)
    print(f"Initial evals: {len(y)} in {time.perf_counter()-t0:.1f}s. Best cross-asset={max(y):.3f}")

    # Surrogate refinement
    N_ROUNDS, TOP_K, N_STARTS = 4, 8, 16
    for r in range(N_ROUNDS):
        surr = fit_surrogate(X, y, steps=1500, lr=1e-3, wd=1e-3, val_frac=0.2, patience=150)
        starts = [np.random.rand(len(ORDER)) for _ in range(N_STARTS - 4)]
        top_idx = np.argsort(y)[-4:]; starts += [X[i] for i in top_idx]
        # explore around surrogate maxima
        cand = []
        for z0 in starts:
            z_star, _ = argmax_on_surrogate(surr, [z0], iters=250, lr=0.05)
            cand.append(z_star)
        # de-dup
        uniq = []
        for z in cand:
            if all(np.linalg.norm(z - u) > 0.05 for u in uniq):
                uniq.append(z)
        Z = np.array(uniq, dtype=np.float32)
        mu, sig = predict_mu_sigma(surr, Z, n=MC_SAMPLES)
        ucb = mu + UCB_BETA * sig
        for idx in np.argsort(ucb)[::-1][:TOP_K]:
            p = decode(Z[idx])
            X.append(encode(p))
            y.append(robust_score_across_assets(assets, p))
        print(f"Round {r+1}: best cross-asset={max(y):.3f}")

    # Final ranking (cross-asset)
    candidates = [decode(x) for x in X]
    val_scores = [robust_score_across_assets(assets, p) for p in candidates]
    top_idxs = np.argsort(val_scores)[::-1][:25]
    top_params = [candidates[i] for i in top_idxs]

    print("\nTop 25 by cross-asset robustness:")
    for rk, i in enumerate(top_idxs, 1):
        print(f"{rk:2d}. {candidates[i]} | score={val_scores[i]:.3f}")

    # Quick check: Sharpe of the #1 set on each asset
    best_p = top_params[0]
    print("\nSharpe of universal best on each asset:")
    for name, df in assets.items():
        s, t, _ = run_bt(df, best_p)
        print(f"{name:7s}  Sharpe={s:6.3f}  TPD={t:.4f}")


if __name__ == '__main__':
     main()