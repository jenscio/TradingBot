# TradingBot

This repository contains a simple Python script implementing a long/short momentum strategy using the Alpaca API. The strategy trades 20 well‑known NASDAQ stocks and goes long the top performers while shorting the weakest.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Alpaca API credentials in the environment:
   ```bash
   export APCA_API_KEY_ID=<your key>
   export APCA_API_SECRET_KEY=<your secret>
   # Optional: custom base URL (defaults to Alpaca's paper trading)
   export APCA_API_BASE_URL=https://paper-api.alpaca.markets
   ```
3. Run the script:
   ```bash
   python long_short_algo.py
   ```

**Note:** This example is for educational purposes. Use at your own risk and customize it to your risk tolerance before trading real funds.

