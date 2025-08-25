import pandas as pd
import numpy as np

# 1. Read CSV without date parsing
full_df = pd.read_csv('data/BATS_QQQ, 60_a45be.csv')
full_df['time'] = pd.to_datetime(full_df['time'].astype(int), unit='s')
full_df.set_index('time', inplace=True)
full_df.sort_index(inplace=True)

# 2. Split into train / test
start_date = pd.Timestamp("2015-11-14") 
split_date = pd.Timestamp("2024-09-24")
end_date   = pd.Timestamp("2025-07-24")
train_df = full_df.loc[(full_df.index >= start_date) & (full_df.index < split_date)].copy()
test_df  = full_df.loc[(full_df.index >= split_date) & (full_df.index <= end_date)].copy() # Testing set

# Debug Print
print("Train:", train_df.index.min(), "→", train_df.index.max(), f"({len(train_df)} rows)")
print("Test: ", test_df.index.min(),  "→", test_df.index.max(),  f"({len(test_df)} rows)")
print(full_df.head())

# 3. Compute moving averages
train_df['EMA_50']  = train_df['close'].ewm(span=50,  adjust=False).mean()
train_df['TrendEMA_100'] = train_df['close'].ewm(span=100, adjust=False).mean()
train_df['EMA_200'] = train_df['close'].ewm(span=200, adjust=False).mean()
train_df['MA_200']  = train_df['close'].rolling(200).mean()

# 4. Compute Ultimate RSI

import pandas as pd

def ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    "Simple wrapper for SMA, EMA, and RMA (Wilder’s MA)."
    ma_type = ma_type.lower()
    if ma_type == 'sma':
        return series.rolling(window=length, min_periods=1).mean()
    elif ma_type == 'ema':
        # pandas span -> alpha = 2/(span+1)
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type in ('rma', 'smma'):  
        # Wilder’s smoothed MA: alpha = 1/length
        return series.ewm(alpha=1/length, adjust=False).mean()
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")

def augmented_rsi(src: pd.Series,
                  length: int,
                  smooth_len: int,
                  ma_type: str = 'ema',
                  smooth_type: str = 'ema') -> (pd.Series, pd.Series):
    """
    Computes the “Ultimate RSI” (from @LuxAlgo on Tradingview)
      - length: lookback for the augmented RSI
      - smooth_len: smoothing for the signal line
      - ma_type: 'sma', 'ema', or 'rma' (Wilder’s)
      - smooth_type: same options for smoothing the RSI into its signal.
    Returns (arsi, signal).
    """
    # 1) rolling high/low
    upper = src.rolling(length, min_periods=1).max()
    lower = src.rolling(length, min_periods=1).min()
    r = upper - lower

    # 2) the “diff” logic
    prev_upper = upper.shift(1)
    prev_lower = lower.shift(1)
    cond_up   = upper > prev_upper
    cond_down = lower < prev_lower

    # if new high, diff = r; if new low, diff = -r; else diff = src.diff()
    diff = pd.Series(src.diff(), index=src.index)
    diff = diff.where(~cond_up & ~cond_down,
                      other=r.where(cond_up, -r))

    # 3) numerator & denominator via our ma() helper
    num = ma(diff, length, ma_type)
    den = ma(diff.abs(), length, ma_type)

    # 4) the ARSI itself
    arsi = num.div(den).mul(50).add(50)

    # 5) signal line
    signal = ma(arsi, smooth_len, smooth_type)

    return arsi, signal

# Example usage on your train_df, by computing 3 RSI's (fast, slow, very slow)
train_df['ARSI_fast'],  train_df['SIG_fast']  = augmented_rsi(
    train_df['close'], length=7,  smooth_len=14, ma_type='rma', smooth_type='ema'
)
train_df['ARSI_slow'],  train_df['SIG_slow']  = augmented_rsi(
    train_df['close'], length=14, smooth_len=14, ma_type='rma', smooth_type='ema'
)
train_df['ARSI_vslow'], train_df['SIG_vslow'] = augmented_rsi(
    train_df['close'], length=50, smooth_len=14, ma_type='rma', smooth_type='ema'
)


# 5. Awesome Oscillator (price momentum)
train_df['AO'] = train_df['high'].add(train_df['low']).div(2).ewm(span=5, adjust=False).mean() \
          - train_df['high'].add(train_df['low']).div(2).ewm(span=34, adjust=False).mean()

# 6. Stdev‑MACD (volatility momentum)

# --- Stdev_fast (length=20) ---
# The direction is now determined by the slope of the SMA(20).
sma_for_direction_fast = train_df['close'].rolling(20).mean()
direction_fast = np.sign(sma_for_direction_fast.diff())
train_df['Stdev_fast'] = train_df['close'].rolling(20).std()*direction_fast
train_df['Stdev_signal'] = train_df['Stdev_fast'].ewm(span=9, adjust=False).mean()


# --- Stdev2_slow (length=200) ---
# The direction is determined by the slope of the SMA(200).
# We already calculated this as 'MA_200', so we can reuse it.
direction_slow = np.sign(train_df['MA_200'].diff())
train_df['Stdev2_slow']   = train_df['close'].rolling(200).std()*direction_slow
train_df['Stdev2_signal'] = train_df['Stdev2_slow'].ewm(span=50, adjust=False).mean()


# 7. Long and short triggers (FINAL FIXED VERSION)

def crossover(series1: pd.Series, series2) -> pd.Series:
    """
    Handles crossover for a Series and another Series OR a constant number.
    """
    # Check if series2 is a number (int or float)
    if isinstance(series2, (int, float)):
        # Logic for crossing over a constant value
        return (series1 > series2) & (series1.shift(1) <= series2)
    else:
        # Original logic for crossing over another Series
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

def crossunder(series1: pd.Series, series2) -> pd.Series:
    """
    Handles crossunder for a Series and another Series OR a constant number.
    """
    # Check if series2 is a number (int or float)
    if isinstance(series2, (int, float)):
        # Logic for crossing under a constant value
        return (series1 < series2) & (series1.shift(1) >= series2)
    else:
        # Original logic for crossing under another Series
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


# --- Now the trigger logic will work perfectly for all cases ---

# Long Triggers
main_long = (train_df['close'] > train_df['MA_200']) & (train_df['MA_200'].diff(3) > 0)
ursi3    = crossover(train_df['ARSI_fast'], train_df['ARSI_slow'])
ursi4    = crossover(train_df['ARSI_vslow'], train_df['SIG_vslow'])
ao_long  = crossover(train_df['AO'], 0) # This will now work correctly
stdev_l  = crossover(train_df['Stdev_fast'], train_df['Stdev_signal'])
stdev2_l = crossover(train_df['Stdev2_slow'], train_df['Stdev2_signal'])



long_raw  = main_long | ursi3 | ursi4 | ao_long | stdev_l | stdev2_l

# Short Triggers
ursi4_s  = crossunder(train_df['ARSI_vslow'], train_df['SIG_vslow'])
ao_short = crossunder(train_df['AO'], 0) # This will now work correctly
ema50_s  = crossunder(train_df['close'], train_df['EMA_50'])
ema200_s = crossunder(train_df['close'], train_df['EMA_200'])


#short_raw = ursi4_s | ao_short | ema50_s | ema200_s
short_raw = ursi4_s | ao_short | ema50_s | ema200_s



# Shift signals
# Shift to next-bar execution without creating NaNs
train_df['LongSignal']  = long_raw.shift(1, fill_value=False).astype(bool)
train_df['ShortSignal'] = short_raw.shift(1, fill_value=False).astype(bool)



# You can now run your backtest loop with this corrected train_df

commission_per_trd = 0.0   # set if you want
risk_frac          = 1.0   # fraction of capital per trade (1.0 = all-in)
allow_shorts       = True  # or False if you want to be realistic first
tick_size = 0.01
slippage_ticks = 1   # to match `slippage=1` in Pine
def fill_price(px, side):  # side: +1 buy, -1 sell
    return px + side * slippage_ticks * tick_size

# BACKTEST lesgoo
capital = 1_000_000
trades = []
position = 0          # +1 long, -1 short, 0 flat
entry_price = None
entry_time  = None
entry_size  = None

trailing_stop_level       = None
stop_loss_pct             = 0.01
take_profit_flip          = True
prev_close = None

# --- equity curve storage ---
equity_times = []
equity_vals  = []
initial_capital = capital  # 1_000_000

for t, row in train_df.iterrows():

    # Mark-to-market equity at the current bar close
    mtm_equity = capital if position == 0 else capital + (row['close'] - entry_price) * entry_size * position
    equity_times.append(t)
    equity_vals.append(mtm_equity)

    # ---------- EXIT / FLIP (OPEN-FIRST) ----------
    if position != 0 and prev_close is not None:
        if position == 1:
            trailing_stop_level = max(trailing_stop_level, prev_close * (1 - stop_loss_pct))
        else:
            trailing_stop_level = min(trailing_stop_level, prev_close * (1 + stop_loss_pct))

    # update now so any `continue` later still keeps it fresh
    prev_close = row['close']
    if position != 0:
        exited = False
        flipped_to = 0
        exit_reason = None

        # 1) Flip/exits that happen at the open
        if take_profit_flip:
            if position == 1 and row['ShortSignal']:
                exited = True
                exit_price = fill_price(row['open'], side=-1)   # sell at open
                exit_reason = 'LONG_FLIP'
                flipped_to = -1
            elif position == -1 and row['LongSignal']:
                exited = True
                exit_price = fill_price(row['open'], side=+1)   # buy-to-cover at open
                exit_reason = 'SHORT_FLIP'
                flipped_to = +1

        # 2) If not flipped at the open, check intrabar trailing stop
        if not exited:
            # gap-through handling (see #2 below)
            if position == 1 and row['low'] <= trailing_stop_level:
                stop_px = trailing_stop_level if row['open'] > trailing_stop_level else row['open']
                exit_price = fill_price(stop_px, side=-1)
                exited = True
                exit_reason = 'LONG_STOP'
            elif position == -1 and row['high'] >= trailing_stop_level:
                stop_px = trailing_stop_level if row['open'] < trailing_stop_level else row['open']
                exit_price = fill_price(stop_px, side=+1)
                exited = True
                exit_reason = 'SHORT_STOP'


        if exited:
            profit = (exit_price - entry_price) * entry_size * position - commission_per_trd
            capital += profit
            trades.append({
                'entry_time': entry_time, 'exit_time': t,
                'entry_price': entry_price, 'exit_price': exit_price,
                'position': position, 'size': entry_size, 'profit': profit,
                'reason': exit_reason, 'capital_after_trade': capital
            })

            # reset
            position = 0
            entry_price = entry_time = entry_size = None
            trailing_stop_level = None

            # If flip requested, open the new side immediately at the SAME open
            if flipped_to != 0:
                if not allow_shorts and flipped_to == -1:
                    # skip opening a short
                    continue
                new_entry_price = fill_price(row['open'], side=+1 if flipped_to==+1 else -1)
                entry_notional = capital * risk_frac
                entry_size = entry_notional / new_entry_price
                capital -= commission_per_trd 
                position   = flipped_to
                entry_price = new_entry_price
                entry_time  = t
                if position == 1:
                    trailing_stop_level = entry_price * (1 - stop_loss_pct)
                    
                else:
                    trailing_stop_level = entry_price * (1 + stop_loss_pct)

            # never re-enter again on this same bar
            continue

    # ---------- ENTRY ----------
    if position == 0:
        if row['LongSignal']:
            new_entry_price = fill_price(row['open'], side=+1)
            entry_notional = capital * risk_frac
            entry_size = entry_notional / new_entry_price
            capital -= commission_per_trd
            position = 1
            entry_price = new_entry_price
            entry_time  = t
            trailing_stop_level = entry_price * (1 - stop_loss_pct)
            

        elif allow_shorts and row['ShortSignal']:
            new_entry_price = fill_price(row['open'], side=-1)
            entry_notional = capital * risk_frac
            entry_size = entry_notional / new_entry_price
            capital -= commission_per_trd
            position = -1
            entry_price = new_entry_price
            entry_time  = t
            trailing_stop_level = entry_price * (1 + stop_loss_pct)

        # If we entered, go to next bar (so we don't update stops intrabar)
        if position != 0:
            continue



# ---------- Close any open position at final bar close ----------
if position != 0:
    last_px = fill_price(train_df['close'].iloc[-1], side=-1 if position==1 else +1)
    last_t  = train_df.index[-1]
    profit  = (last_px - entry_price) * entry_size * position - commission_per_trd
    capital += profit
    trades.append({
        'entry_time': entry_time, 'exit_time': last_t,
        'entry_price': entry_price, 'exit_price': last_px,
        'position': position, 'size': entry_size, 'profit': profit,
        'reason': 'FINAL_MARK', 'capital_after_trade': capital
    })

# 5. Build trades DataFrame and compute P&L
trades_df = pd.DataFrame(trades)
# --- Trade counters & win rate ---
total_trades  = len(trades_df)
long_trades   = (trades_df['position'] == 1).sum()
short_trades  = (trades_df['position'] == -1).sum()
wins          = (trades_df['profit'] > 0).sum()
losses        = (trades_df['profit'] <= 0).sum()
win_rate      = wins / max(wins + losses, 1)

# Trades per year (rough)
span_years = max((train_df.index[-1] - train_df.index[0]).days / 365.25, 1e-9)
tpy = total_trades / span_years

print(f"\nTrades: {total_trades} (long={long_trades}, short={short_trades})")
print(f"Win rate: {win_rate:.2%}   |   Wins={wins}, Losses={losses}")
print(f"Trades per year (approx): {tpy:.2f}")

print(trades_df[['entry_time','exit_time','entry_price','exit_price',
                 'position','size','profit','capital_after_trade']])
print(f"\nStarting capital:  $1,000,000.00")
print(f"Final capital:     ${capital:,.2f}")
print(f"Total P&L:         ${capital - 1_000_000:,.2f}")
print(f"Total Return: {(capital/1_000_000 - 1):.2%}")

# ----- Build equity curve & cumulative returns -----
equity = pd.Series(equity_vals, index=pd.Index(equity_times, name="time"))
equity = equity.reindex(train_df.index, method="pad")  # align to price index

strat_ret = equity / initial_capital - 1.0                  # strategy cumulative return
px_ret    = train_df['close'] / train_df['close'].iloc[0] - 1.0  # QQQ cumulative return

# ----- Plot both on one chart in % -----
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(strat_ret.index, strat_ret, label="Strategy", color="tab:blue")
ax.plot(px_ret.index,    px_ret,    label="QQQ",      color="tab:orange")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_title("Cumulative Returns (%)")
ax.set_xlabel("Time")
ax.set_ylabel("Return")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


       

