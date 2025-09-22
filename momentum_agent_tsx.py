import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# --------------------------
# CONFIG
# --------------------------
TICKERS = ["RY.TO", "TD.TO", "BNS.TO", "ENB.TO", "TRP.TO"]  # sample list
START = "2018-01-01"
END = datetime.today().strftime("%Y-%m-%d")
LOOKBACK_MONTHS = 6         # momentum lookback
REBALANCE_DAYS = 21         # roughly monthly trading days
TOP_N = 3
INITIAL_CAPITAL = 100_000
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade (example)
SLIPPAGE_PCT = 0.0005

# --------------------------
# Data fetch
# --------------------------
def fetch_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
    # yfinance returns different structure for single vs many tickers - handle both
    # We'll build a panel: prices[ticker] = close series
    close = pd.DataFrame()
    for t in tickers:
        try:
            close[t] = data[t]['Close']
        except Exception:
            # fallback if data has single-level
            if 'Close' in data:
                close[t] = data['Close']
            else:
                print("No data for", t)
    close = close.dropna(how='all')
    return close

prices = fetch_prices(TICKERS, START, END).ffill().dropna(axis=0, how='all')

# --------------------------
# Feature: momentum
# --------------------------
def compute_momentum(prices, lookback_months):
    lookback_days = int(21 * lookback_months)
    returns = prices.pct_change(periods=lookback_days)
    return returns

momentum = compute_momentum(prices, LOOKBACK_MONTHS)

# --------------------------
# Backtest: monthly rebalance to top-N momentum
# --------------------------
def backtest_momentum(prices, momentum, top_n, rebalance_days, initial_capital):
    dates = prices.index
    cash = initial_capital
    positions = {t: 0 for t in prices.columns}
    portfolio_value = []
    last_rebalance = dates[0]
    for i, date in enumerate(dates):
        # rebalance on the first trading day and every rebalance_days
        if (i == 0) or ((i - dates.get_loc(last_rebalance)) % rebalance_days == 0):
            last_rebalance = date
            # choose top_n stocks by momentum as of this date (if available)
            if date in momentum.index:
                scores = momentum.loc[date].dropna()
                if len(scores) >= top_n:
                    picks = scores.sort_values(ascending=False).iloc[:top_n].index.tolist()
                else:
                    picks = scores.sort_values(ascending=False).index.tolist()
            else:
                picks = []

            # liquidate current positions
            total_value = cash + sum(positions[t] * prices.at[date, t] for t in prices.columns)
            cash = total_value
            for t in positions:
                positions[t] = 0

            # equally allocate cash to picks (simple)
            if picks:
                weight_cash = cash / len(picks)
                for t in picks:
                    price = prices.at[date, t]
                    if np.isnan(price) or price <= 0:
                        continue
                    # buy shares (floor)
                    shares = int((weight_cash * (1 - TRANSACTION_COST_PCT)) / price)
                    positions[t] = shares
                # update cash after buys
                cash = cash - sum(positions[t] * prices.at[date, t] for t in prices.columns)
                # subtract approximate transaction costs
                cash = cash - TRANSACTION_COST_PCT * (total_value - cash)

        # daily portfolio value
        pv = cash + sum(positions[t] * prices.at[date, t] for t in prices.columns)
        portfolio_value.append(pv)

    pf = pd.Series(portfolio_value, index=dates)
    return pf

pf_series = backtest_momentum(prices, momentum, TOP_N, REBALANCE_DAYS, INITIAL_CAPITAL)

# --------------------------
# Simple performance metrics
# --------------------------
def performance_stats(pf):
    daily_rets = pf.pct_change().dropna()
    total_return = (pf.iloc[-1] / pf.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(daily_rets)) - 1 if len(daily_rets)>0 else np.nan
    annualized_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (annualized_return / annualized_vol) if annualized_vol > 0 else np.nan
    max_dd = (pf.cummax() - pf).max() / pf.cummax().max()
    return {
        "start_value": pf.iloc[0],
        "end_value": pf.iloc[-1],
        "total_return": total_return,
        "ann_return": annualized_return,
        "ann_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }

stats = performance_stats(pf_series)

print("Performance stats:", stats)

# --------------------------
# Plot equity curve
# --------------------------
plt.figure(figsize=(10,5))
plt.plot(pf_series)
plt.title("Momentum Agent — Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)
plt.show()