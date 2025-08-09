import yfinance as yf
import pandas as pd
import numpy as np
import ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def sma(series, length):
    s = pd.Series(series)
    return s.rolling(length).mean()

def ema(series, span):
    s = pd.Series(series)
    return s.ewm(span=span, adjust=False).mean()

def rma(series, length):
    # Wilder's smoothing: alpha = 1/length
    s = pd.Series(series)
    return s.ewm(alpha=1/length, adjust=False).mean()

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
    """Returns (arsi, signal). Matches your pandas implementation."""
    src = pd.Series(src)

    upper = src.rolling(length, min_periods=1).max()
    lower = src.rolling(length, min_periods=1).min()
    r = upper - lower

    prev_upper = upper.shift(1)
    prev_lower = lower.shift(1)
    cond_up   = upper > prev_upper
    cond_down = lower < prev_lower

    diff = src.diff()
    diff = diff.where(~cond_up & ~cond_down, other=r.where(cond_up, -r))

    num = ma(diff, length, ma_type)
    den = ma(diff.abs(), length, ma_type)

    arsi = (num / den) * 50 + 50
    signal = ma(arsi, smooth_len, smooth_type)
    return arsi, signal

def awesome_osc(high, low):
    """Your AO: EMA(5, median) - EMA(34, median)"""
    h = pd.Series(high); l = pd.Series(low)
    med = (h + l) / 2.0
    return med.ewm(span=5, adjust=False).mean() - med.ewm(span=34, adjust=False).mean()

def stdev_fast_block(close):
    """Stdev_fast and its signal (length=20, signal EMA 9). Direction = sign(SMA20 slope)."""
    c = pd.Series(close)
    sma20 = c.rolling(20).mean()
    direction = np.sign(sma20.diff())
    stdev_fast = c.rolling(20).std() * direction
    stdev_signal = stdev_fast.ewm(span=9, adjust=False).mean()
    return stdev_fast, stdev_signal

def stdev2_slow_block(close):
    """Stdev2_slow and its signal (length=200, signal EMA 50). Direction = sign(SMA200 slope)."""
    c = pd.Series(close)
    ma200 = c.rolling(200).mean()
    direction = np.sign(ma200.diff())
    stdev2_slow = c.rolling(200).std() * direction
    stdev2_signal = stdev2_slow.ewm(span=50, adjust=False).mean()
    return stdev2_slow, stdev2_signal

class ARSIstrat(Strategy):
    sl_pct = 0.01   # Stop Loss
    tp_pct = None   # Take Profit
    n_fast = 7
    n_slow = 14
    n_vslow = 50
    sig_len = 14

    def init(self):
        
        c = self.data.Close
        h = self.data.High
        l = self.data.Low

        # Moving averages
        self.EMA_50        = self.I(ema, c, 50)
        self.TrendEMA_100  = self.I(ema, c, 100)
        self.EMA_200       = self.I(ema, c, 200)
        self.MA_200        = self.I(sma, c, 200)

        # Ultimate/augmented RSI pack (fast, slow, very slow)
        self.ARSI_fast,  self.SIG_fast  = self.I(
            augmented_rsi, c, self.n_fast, self.sig_len, 'rma', 'ema'
        )
        self.ARSI_slow,  self.SIG_slow  = self.I(
            augmented_rsi, c, self.n_slow, self.sig_len, 'rma', 'ema'
        )
        self.ARSI_vslow, self.SIG_vslow = self.I(
            augmented_rsi, c, self.n_vslow, self.sig_len, 'rma', 'ema'
        )

        # Awesome Oscillator (your EMA version)
        self.AO = self.I(awesome_osc, h, l)

        # Stdev-MACD style blocks
        self.Stdev_fast,  self.Stdev_signal  = self.I(stdev_fast_block, c)
        self.Stdev2_slow, self.Stdev2_signal = self.I(stdev2_slow_block, c)
    
    def next(self):

        if len(self.data) < 205:
            return

        main_long = (self.data.Close[-1] > self.MA_200[-1]) and (self.MA_200[-1] > self.MA_200[-4])
        ursi3    = crossover(self.ARSI_fast,  self.ARSI_slow)
        ursi4    = crossover(self.ARSI_vslow, self.SIG_vslow)
        stdev_l  = crossover(self.Stdev_fast,  self.Stdev_signal)
        stdev2_l = crossover(self.Stdev2_slow, self.Stdev2_signal)

        # B) Tiny wrappers (clear + fast)
        def xover_scalar(arr, lvl):  a = np.asarray(arr); return a[-1] > lvl and a[-2] <= lvl
        def xunder_scalar(arr, lvl): a = np.asarray(arr); return a[-1] < lvl and a[-2] >= lvl
        ao_long  = xover_scalar(self.AO, 0.0)
        ao_short = xunder_scalar(self.AO, 0.0)

        ema50_s  = crossover(self.EMA_50, self.data.Close)
        ema200_s = crossover(self.EMA_200, self.data.Close)

        long_raw  = main_long or ursi3 or ursi4 or ao_long or stdev_l or stdev2_l
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


# 1. Read CSV without date parsing
full_df = pd.read_csv('BATS_QQQ, 60_a45be.csv')
full_df['time'] = pd.to_datetime(full_df['time'].astype(int), unit='s')
full_df.set_index('time', inplace=True)
full_df.sort_index(inplace=True)
full_df = full_df.rename(columns={
    'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'
})
print(full_df.head())

bt = Backtest(full_df, ARSIstrat, cash=1000000, commission=0.0,spread=0.0001)

stats = bt.run()
print(stats)
bt.plot() 