The strategy is a rule-based trading system built on augmented RSI signals, trend filters
and volatility blocks. It combines three ARSI regimes (fast, slow, very slow), the Awesome Oscillator, 
and two stdev “MACD-style” blocks, with long-term EMAs/SMA200 as filters. 
Entries occur on crossovers from these signals: longs when price is trending above SMA200 
and momentum/volatility confirm, shorts on the opposite. Positions are protected with a fixed stop-loss (sl_pct) 
and flip when an opposite impulse appears. 

Training is done on 2014–2020 QQQ data, optimizing with a robust composite score 
(Sharpe, Sortino, Calmar minus drawdown/tail risk, scaled by trade count). 
Top parameter sets are then validated on 2021–2022 out-of-sample, with the best candidates held for final testing on 2023+ data. 
It is designed to favor robust, stable performance over fragile curve-fits, on different ETF's like the QQQ, SP500, MSCI World, DAX, SMI, CAC40.

To check for overfitting risk, we also run Monte Carlo permutations (resampling and shuffling of trade sequences and returns). 
This stress-tests the stability of the strategy’s edge and ensures that reported results are not artifacts of 
specific trade orderings or favorable market regimes.