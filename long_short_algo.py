import os
import alpaca_trade_api as tradeapi
import pandas as pd
from typing import List

API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

tickers: List[str] = [
    'AAPL','MSFT','AMZN','GOOGL','GOOG','META','TSLA','NVDA','PYPL','ADBE',
    'INTU','CMCSA','PEP','CSCO','AVGO','COST','AMGN','TXN','QCOM','SBUX'
]

LONG_SHORT_COUNT = 5  # number of stocks to long and short
NOTIONAL_PER_TRADE = 5000  # USD notional value per position

def get_api():
    if not API_KEY or not API_SECRET:
        raise ValueError('Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.')
    return tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def fetch_price_history(api, symbols: List[str], lookback: int = 20) -> pd.DataFrame:
    barset = api.get_bars(symbols, '1Day', limit=lookback)
    data = {}
    for symbol in symbols:
        bars = [bar.c for bar in barset[symbol]]
        if len(bars) < lookback:
            raise ValueError(f"Not enough data for {symbol}")
        data[symbol] = bars
    df = pd.DataFrame(data)
    return df

def compute_returns(df: pd.DataFrame) -> pd.Series:
    returns = df.iloc[-1] / df.iloc[0] - 1
    return returns.sort_values(ascending=False)

def rebalance(api, returns: pd.Series):
    longs = returns.head(LONG_SHORT_COUNT).index
    shorts = returns.tail(LONG_SHORT_COUNT).index
    to_trade = list(longs) + list(shorts)
    positions = {p.symbol: p for p in api.list_positions()}

    # Close positions not in target universe
    for symbol, pos in positions.items():
        if symbol not in to_trade:
            print(f"Closing position in {symbol}")
            side = 'sell' if pos.side == 'long' else 'buy'
            api.submit_order(symbol, qty=abs(int(float(pos.qty))), side=side, type='market', time_in_force='day')

    for symbol in longs:
        qty = int(NOTIONAL_PER_TRADE / float(api.get_latest_trade(symbol).p))
        if symbol in positions and positions[symbol].side == 'long':
            continue
        if symbol in positions and positions[symbol].side == 'short':
            api.submit_order(symbol, qty=abs(int(float(positions[symbol].qty))), side='buy', type='market', time_in_force='day')
        print(f"Going long {symbol}")
        api.submit_order(symbol, qty=qty, side='buy', type='market', time_in_force='day')

    for symbol in shorts:
        qty = int(NOTIONAL_PER_TRADE / float(api.get_latest_trade(symbol).p))
        if symbol in positions and positions[symbol].side == 'short':
            continue
        if symbol in positions and positions[symbol].side == 'long':
            api.submit_order(symbol, qty=abs(int(float(positions[symbol].qty))), side='sell', type='market', time_in_force='day')
        print(f"Going short {symbol}")
        api.submit_order(symbol, qty=qty, side='sell', type='market', time_in_force='day')

def main():
    api = get_api()
    df = fetch_price_history(api, tickers)
    returns = compute_returns(df)
    rebalance(api, returns)

if __name__ == '__main__':
    main()

