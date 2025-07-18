from alpaca.trading.client import TradingClient as TC
import alpaca_trade_api
import pandas as pd

ALPACA_API_KEY = "PKZONSLMACOGACTKXDEE"
ALPACA_SECRET_KEY = "SdmsscHFJmnduuqGKDUUeFKewsX1fRnY285IzFt4"

print("All good. Alpaca and Pandas imported successfully.")
trading_client = TC(ALPACA_API_KEY,ALPACA_SECRET_KEY)
account = trading_client.get_account()
print("Account Number:", account.account_number)
print("Buying Power:", account.buying_power)