from alpaca.trading.client import TradingClient as TC
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce 
import alpaca_trade_api
import pandas as pd


ALPACA_API_KEY = "PKZONSLMACOGACTKXDEE"
ALPACA_SECRET_KEY = "SdmsscHFJmnduuqGKDUUeFKewsX1fRnY285IzFt4"

print("All good. Alpaca and Pandas imported successfully.")
trading_client = TC(ALPACA_API_KEY,ALPACA_SECRET_KEY)
account = trading_client.get_account()
print("Account Number:", account.account_number)
print("Buying Power:", account.buying_power)

market_orders = MarketOrderRequest(
    symbol = "SPY",
    qty = 1,
    side = OrderSide.BUY,
    time_in_force = TimeInForce.DAY
 )

market_order = trading_client.submit_order(market_orders)
print(market_order)

limit_order1 = LimitOrderRequest(
    symbol = "SPY",
    qty = 1,
    side = OrderSide.BUY,
    time_in_force = TimeInForce.DAY ,
    limit_price = 628
)
limit_order2 = LimitOrderRequest(
    symbol = "SPY",
    qty = 1,
    side = OrderSide.BUY,
    time_in_force = TimeInForce.DAY ,
    limit_price = 400
) 

limit_order = trading_client.submit_order(limit_order1)
limit_order = trading_client.submit_order(limit_order2)

from alpaca.trading.requests import GetOrdersRequest
from  alpaca.trading.enums import QueryOrderStatus

request_params = GetOrdersRequest(
    status = QueryOrderStatus.OPEN,

)

orders = trading_client.get_orders(request_params)

for order in orders:
    print(order.id)
