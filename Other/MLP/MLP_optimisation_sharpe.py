# optimize_sharpe_surrogate.py
import os
import time
import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
import yfinance as yf

# Backtesting
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Your strategy class lives here
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
from Strategy.strategy import ARSIstrat

# ============================= Config =========================================
CUTOFF = "2022-01-01"

ASSETS = {
    "QQQ": ("csv", os.path.join("CSV_files", "BATS_QQQ, 60_a45be.csv"),
             {"start": "2014-12-17", "end_excl": CUTOFF}),
}

# ========================= Data loading stuff =================================
def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten yfinance multi-index if present
    if isinstance(df.columns, pd.MultiIndex):
        fields = {'Open', 'High', 'Low', 'Close', 'Volume'}
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        if fields.issubset(lvl0):
            df = df.droplevel(1, axis=1)
        elif fields.issubset(lvl1):
            df = df.droplevel(0, axis=1)
        else:
            raise ValueError("Can't find OHLCV fields in MultiIndex columns")
    if 'Adj Close' in df.columns:
        df = df.drop(columns=['Adj Close'])
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[cols].astype(float)

def load_data(csv_path: str, start='2014-12-17', end_excl='2022-01-01'):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'].astype(int), unit='s')
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    df = df.loc[(df.index >= start) & (df.index < end_excl)].copy()
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volume': 'Volume'})
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.dropna(subset=['Open','High','Low','Close'])
    return df

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
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    return df

# In case we want to add more assets
def load_all_assets():
    out = {}
    for name, (kind, src, kw) in ASSETS.items():
        df = load_asset(name, kind, src, kw)
        df = df.dropna(subset=["Open","High","Low","Close"]).copy()
        out[name] = df
    return out

# ======================== Param space + helpers ===============================
SPACE = {
    'sl_pct': (0.001, 0.02, 'float'),  # 0.1% .. 2%
    'n_fast': (5, 25, 'int'),
    'n_slow': (10, 55, 'int'),
    'n_vslow': (30, 120, 'int'),
    'sig_len': (5, 30, 'int'),
}
ORDER = list(SPACE.keys())

# Gaps andordering constraints
MIN_GAP_FS = 3   # n_slow - n_fast  >= 3
MIN_GAP_SV = 7   # n_vslow - n_slow >= 7

def _enforce_order_lt(p):
    """Clamp to bounds and enforce n_fast < n_slow < n_vslow with minimum gaps."""
    f_lo, f_hi, _ = SPACE['n_fast']
    s_lo, s_hi, _ = SPACE['n_slow']
    v_lo, v_hi, _ = SPACE['n_vslow']

    nf = int(np.clip(int(round(p['n_fast'])), f_lo, f_hi))

    ns_min = max(s_lo, nf + MIN_GAP_FS)
    ns = int(np.clip(int(round(p['n_slow'])), ns_min, s_hi))
    nf = min(nf, ns - MIN_GAP_FS)

    nv_min = max(v_lo, ns + MIN_GAP_SV)
    nv = int(np.clip(int(round(p['n_vslow'])), nv_min, v_hi))
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

def sample_params(n):
    Ps = []
    f_lo, f_hi, _ = SPACE['n_fast']
    s_lo, s_hi, _ = SPACE['n_slow']
    v_lo, v_hi, _ = SPACE['n_vslow']
    
    nf_max = min(f_hi, s_hi - MIN_GAP_FS, v_hi - (MIN_GAP_FS + MIN_GAP_SV))
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

# ======================== Strategy wiring + backtest ==========================
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
    bt = Backtest(data, Strat, cash=1_000_000, commission=0.0,
                  spread=spread, finalize_trades=True)
    stats = bt.run()
    sharpe = float(stats.get('Sharpe Ratio', 0.0))
    # turnover proxy (kept for diagnostics if you want it)
    trades = int(getattr(stats, '_trades', pd.DataFrame()).shape[0]) if hasattr(stats, '_trades') else 0
    days = max(1, (data.index[-1] - data.index[0]).days)
    tpd = trades / days
    return sharpe, tpd, stats

# ============================ Surrogate (Torch) ===============================
try:
    import torch
    import torch.nn as nn
except Exception:
    raise SystemExit("PyTorch not installed. `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`")

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


def fit_surrogate(X, y, steps=2500, lr=3e-4, wd=1e-3, val_frac=0.2, patience=250,
                  width=256, depth=8, p_dropout=0.10, clip=1.0):
    from copy import deepcopy
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.float32)

    # Standardize X and y for easier training
    Xm, Xs = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    ym, ys = y.mean(), y.std() + 1e-8
    Xn, yn = (X - Xm) / Xs, (y - ym) / ys

    n = len(Xn)
    if n < 10: raise ValueError("Need at least 10 samples to fit surrogate")
    idx = np.random.permutation(n)
    k = int(n * (1 - val_frac))
    tr, va = idx[:k], idx[k:]

    Xt = torch.tensor(Xn[tr]); yt = torch.tensor(yn[tr])
    Xv = torch.tensor(Xn[va]); yv = torch.tensor(yn[va])

    model = Surrogate(X.shape[1], width=width, depth=depth, p_dropout=p_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss = nn.MSELoss()

    best = (1e9, None)
    bad = 0
    for _ in range(steps):
        model.train(); opt.zero_grad(set_to_none=True)
        pred = model(Xt); l = loss(pred, yt)
        l.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        opt.step(); sched.step()

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

    # attach scalers for downstream use
    model._scalers = (torch.tensor(Xm), torch.tensor(Xs), float(ym), float(ys))
    return model


# --- Fix argmax_on_surrogate to feed normalized inputs ---
def argmax_on_surrogate(model, starts, iters=250, lr=0.05):
    best_z, best_pred = None, -1e9
    Xm, Xs, ym, ys = model._scalers  # normalization the model expects
    for z0 in starts:
        z = torch.tensor(z0.copy(), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            z_c = torch.clamp(z, 0.0, 1.0)
            z_in = (z_c - Xm) / Xs
            pred_norm = model(z_in)               # model predicts normalized y
            loss = -pred_norm                     # maximizing normalized y is fine
            loss.backward(); opt.step()
        with torch.no_grad():
            z_c = torch.clamp(z, 0.0, 1.0)
            pred = _model_predict_raw(model, z_c[None, :].cpu().numpy())[0]  # de-normalized score
            if pred > best_pred:
                best_pred, best_z = pred, z_c.detach().numpy()
    return best_z, best_pred


# MC-Dropout uncertainty for UCB
UCB_BETA = 0.5     # explore weight; try 0.3–1.0
MC_SAMPLES = 64    # dropout passes

def predict_mu_sigma(model, Z, n=MC_SAMPLES):
    """
    MC-dropout around the fitted surrogate, with proper input standardization
    and output de-standardization.
    """
    model.train()  # enable dropout
    Xm, Xs, ym, ys = model._scalers
    Xz = (torch.tensor(Z, dtype=torch.float32) - Xm) / Xs
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(model(Xz).cpu().numpy())
    P = np.stack(preds, 0) * ys + ym
    return P.mean(0), P.std(0)

def fit_surrogate_ensemble(X, y, n_models=5, **kw):
    models = []
    for i in range(n_models):
        torch.manual_seed(1000 + i)
        np.random.seed(1000 + i)
        models.append(fit_surrogate(X, y, **kw))
    return models

def predict_mu_sigma_ens(models, Z, mc_samples=0):
    # plain ensemble mean/std
    preds = []
    for m in models:
        preds.append(_model_predict_raw(m, Z))
    P = np.stack(preds, 0)  # [n_models, n_points]
    mu = P.mean(0); sig = P.std(0)

    # optional MC-dropout on each model for extra epistemic+aleatoric
    if mc_samples and mc_samples > 0:
        more = []
        for m in models:
            m.train()  # enable dropout
            Xm, Xs, ym, ys = m._scalers
            Xz = (torch.tensor(Z, dtype=torch.float32) - Xm) / Xs
            with torch.no_grad():
                S = []
                for _ in range(mc_samples):
                    S.append(m(Xz).cpu().numpy())
            S = np.stack(S, 0) * ys + ym
            more.append(S)  # [mc, n_points]
        M = np.stack(more, 0)  # [n_models, mc, n_points]
        mu = M.mean((0,1))
        sig = M.std((0,1))
    return mu, sig

def sobol_unit(n, d, seed=123):
    engine = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed)
    return engine.draw(n).numpy().astype(np.float32)

def _model_predict_raw(model, Z):
    Xm, Xs, ym, ys = model._scalers
    Xz = (torch.tensor(Z, dtype=torch.float32) - Xm) / Xs
    with torch.no_grad():
        pred_n = model(Xz).cpu().numpy()
    return pred_n * ys + ym

# ============================ Objective =======================================
def sharpe_across_assets(assets_dict, p):
    """Mean Sharpe across assets (single asset => just that Sharpe)."""
    sharpes = []
    for name, df in assets_dict.items():
        s, _, _ = run_bt(df, p)  # s is Sharpe Ratio
        sharpes.append(float(s))
    if not sharpes:
        return -1e9
    return float(np.mean(sharpes))

# =============================== Main =========================================
def main():
    # Repro seeds
    np.random.seed(22)
    random.seed(22)
    try:
        torch.manual_seed(22)
    except Exception:
        pass

    # Load assets
    assets = load_all_assets()
    print({k: (assets[k].index.min(), assets[k].index.max(), len(assets[k])) for k in assets})

    # 1) Initial random design
    N_INIT = 60
    Z0 = sobol_unit(N_INIT, len(ORDER), seed=12)
    pool = [decode(z) for z in Z0]
    X, y = [], []
    t0 = time.perf_counter()
    for p in pool:
        s_train = sharpe_across_assets(assets, p)
        X.append(encode(p)); y.append(s_train)
    print(f"Initial evals: {len(y)} in {time.perf_counter()-t0:.1f}s. Best Sharpe={max(y):.3f}")

    # 2) Surrogate refinement
    N_ROUNDS, TOP_K, N_STARTS = 6, 12, 32
    for r in range(N_ROUNDS):
        surrs = fit_surrogate_ensemble(X, y,n_models=5,steps=2200, lr=8e-4, wd=1e-3, val_frac=0.2, patience=220,width=128, depth=6, p_dropout=0.10)

        # starts: random + the current top-4
        starts = [np.random.rand(len(ORDER)) for _ in range(N_STARTS - 4)]
        top_idx = np.argsort(y)[-4:]; starts += [X[i] for i in top_idx]

        # hill-climb on surrogate from each start
        cand = []
        for z0 in starts:
            z_star, _ = argmax_on_surrogate(surrs[0], [z0], iters=400, lr=0.05)
            cand.append(z_star)

        # de-dup similar points
        uniq = []
        for z in cand:
            if all(np.linalg.norm(z - u) > 0.05 for u in uniq):
                uniq.append(z)

        # UCB pick TOP_K to evaluate for real
        Z = np.array(uniq, dtype=np.float32)
        mu, sig = predict_mu_sigma_ens(surrs, Z, mc_samples=16)
        ucb = mu + UCB_BETA * sig

        added = 0
        for idx in np.argsort(ucb)[::-1][:TOP_K]:
            p = decode(Z[idx])
            X.append(encode(p))
            y.append(sharpe_across_assets(assets, p))
            added += 1

        print(f"Round {r+1}: added {added}, best Sharpe={max(y):.3f}")

    # 3) Final ranking by Sharpe (no CV, just objective)
    candidates = [decode(x) for x in X]
    val_scores = [sharpe_across_assets(assets, p) for p in candidates]
    idxs = np.argsort(val_scores)[::-1]
    K_FINAL = min(25, len(idxs))
    finalists = [candidates[i] for i in idxs[:K_FINAL]]

    print("\nTop 25 by Sharpe:")
    for i, idx in enumerate(idxs[:K_FINAL], 1):
        print(f"{i:2d}. {candidates[idx]} | Sharpe={val_scores[idx]:.3f}")

    # 4) Quick per-asset check for the #1 set
    best_p = finalists[0]
    print("\nSharpe of best on each asset:")
    for name, df in assets.items():
        s, t, _ = run_bt(df, best_p)
        print(f"{name:7s}  Sharpe={s:6.3f}  TPD={t:.4f}")

if __name__ == '__main__':
    main()
